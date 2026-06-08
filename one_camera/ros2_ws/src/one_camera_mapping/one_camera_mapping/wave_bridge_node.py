"""Wave arayuzu kopru node'u.

/map (OccupancyGrid) ve /detections_3d (Detection3DArray) tuketir,
'wave screen.html' beklentisi olan signal_field formatina cevirip
WebSocket (ws://0.0.0.0:8765/ws/sensing) ile yayinlar.
"""

import asyncio
import json
import os
import threading

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray

import websockets

# Paylasilan durum (WebSocket thread <-> ROS thread)
GRID_COLS = 20
GRID_ROWS = 20
WS_PORT = int(os.environ.get("WAVE_BRIDGE_PORT", "8765"))
_latest_field = [0.0] * (GRID_COLS * GRID_ROWS)
_latest_detections = []


def _build_payload():
    return json.dumps({
        "signal_field": {
            "grid_size": [GRID_COLS, 1, GRID_ROWS],
            "values": list(_latest_field),
        },
        "detections": list(_latest_detections),
    })


async def _ws_handler(websocket):
    try:
        path = websocket.request.path
    except AttributeError:
        path = getattr(websocket, "path", "/ws/sensing")
    if path != "/ws/sensing":
        await websocket.close(1008, "Invalid path")
        return
    try:
        while True:
            await websocket.send(_build_payload())
            await asyncio.sleep(0.1)
    except Exception:
        pass


async def _ws_main():
    async with websockets.serve(_ws_handler, "0.0.0.0", WS_PORT):
        print(f"Wave bridge WebSocket: ws://0.0.0.0:{WS_PORT}/ws/sensing")
        await asyncio.Future()


def _run_websocket_server():
    asyncio.run(_ws_main())


class WaveBridgeNode(Node):
    def __init__(self):
        super().__init__("wave_bridge_node")

        self.declare_parameter("grid_cols", GRID_COLS)
        self.declare_parameter("grid_rows", GRID_ROWS)

        self.cols = int(self.get_parameter("grid_cols").value)
        self.rows = int(self.get_parameter("grid_rows").value)

        self.create_subscription(OccupancyGrid, "/map", self._on_map, 1)
        self.create_subscription(
            Detection3DArray, "/detections_3d", self._on_detections, 10
        )
        self.get_logger().info(
            f"wave_bridge ROS dinliyor. ws://0.0.0.0:{WS_PORT}/ws/sensing "
            f"grid {self.cols}x{self.rows}"
        )

    def _on_map(self, msg):
        global _latest_field
        data = np.array(msg.data, dtype=np.float32).reshape(
            msg.info.height, msg.info.width
        )
        data[data < 0] = 0.0
        gh = max(1, msg.info.height // self.rows)
        gw = max(1, msg.info.width // self.cols)
        field = [0.0] * (self.cols * self.rows)
        for r in range(self.rows):
            for c in range(self.cols):
                y0, y1 = r * gh, min(msg.info.height, (r + 1) * gh)
                x0, x1 = c * gw, min(msg.info.width, (c + 1) * gw)
                block = data[y0:y1, x0:x1]
                val = float(block.max()) / 100.0 if block.size else 0.0
                field[r * self.cols + c] = max(0.0, min(1.0, val))
        _latest_field = field

    def _on_detections(self, msg):
        global _latest_detections
        dets = []
        for det in msg.detections:
            if not det.results:
                continue
            p = det.bbox.center.position
            dets.append({
                "label": det.results[0].hypothesis.class_id,
                "score": round(float(det.results[0].hypothesis.score), 3),
                "x": round(float(p.x), 3),
                "y": round(float(p.y), 3),
                "z": round(float(p.z), 3),
            })
        _latest_detections = dets


def main(args=None):
    # WebSocket ONCE rclpy'den once baslatilir (rclpy/websockets cakismasini onler)
    threading.Thread(target=_run_websocket_server, daemon=True).start()

    rclpy.init(args=args)
    node = WaveBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
