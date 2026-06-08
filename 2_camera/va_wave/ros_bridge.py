"""ROS /va_wave/stereo_point -> Wave arayuzu WebSocket bridge."""

import asyncio
import json
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import websockets

latest_point = {"x": 0.0, "y": 0.0, "z": 0.0}
clients = set()


def build_signal_field(x, y, z):
    cols, rows = 20, 20
    values = [0.0] * (cols * rows)

    if z <= 0:
        return {"signal_field": {"grid_size": [cols, 1, rows], "values": values}}

    cx = int(max(0, min(cols - 1, (x + 1.0) / 2.0 * (cols - 1))))
    cy = int(max(0, min(rows - 1, (y + 1.0) / 2.0 * (rows - 1))))
    intensity = min(1.0, 1.5 / max(z, 0.1))

    for r in range(rows):
        for c in range(cols):
            dist = ((c - cx) ** 2 + (r - cy) ** 2) ** 0.5
            values[r * cols + c] = intensity * max(0.0, 1.0 - dist / 4.0)

    return {
        "signal_field": {"grid_size": [cols, 1, rows], "values": values},
        "coordinates": {"x": x, "y": y, "z": z},
    }


async def ws_handler(websocket):
    if websocket.request.path != "/ws/sensing":
        await websocket.close(1008, "Invalid path")
        return

    clients.add(websocket)
    try:
        while True:
            payload = build_signal_field(
                latest_point["x"], latest_point["y"], latest_point["z"]
            )
            await websocket.send(json.dumps(payload))
            await asyncio.sleep(0.1)
    finally:
        clients.discard(websocket)


async def ws_main():
    async with websockets.serve(ws_handler, "0.0.0.0", 8765):
        print("Wave bridge WebSocket: ws://0.0.0.0:8765/ws/sensing")
        await asyncio.Future()


def run_websocket_server():
    asyncio.run(ws_main())


class BridgeNode(Node):
    def __init__(self):
        super().__init__("ros_bridge")
        self.create_subscription(
            Float32MultiArray,
            "/va_wave/stereo_point",
            self.callback,
            10,
        )
        self.get_logger().info("ROS bridge dinliyor: /va_wave/stereo_point")

    def callback(self, msg):
        if len(msg.data) >= 3:
            latest_point["x"] = float(msg.data[0])
            latest_point["y"] = float(msg.data[1])
            latest_point["z"] = float(msg.data[2])
            self.get_logger().info(
                f"Bridge -> X:{latest_point['x']:.3f} "
                f"Y:{latest_point['y']:.3f} Z:{latest_point['z']:.3f}"
            )


def main(args=None):
    threading.Thread(target=run_websocket_server, daemon=True).start()

    rclpy.init(args=args)
    node = BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
