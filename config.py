# Network configuration — update PI_HOST and LAPTOP_HOST to match your network.

# Raspberry Pi's IP address
PI_HOST = '192.168.43.255'

# Laptop's IP address (runs training data collection and streaming server)
LAPTOP_HOST = '192.168.43.224'

# Port the Pi's local socket binds to for streaming
PI_STREAM_PORT = 7000

# Port the laptop's streaming server listens on
LAPTOP_STREAM_PORT = 8000

# Port the laptop's data collection server listens on
COLLECT_PORT = 8014

# Port the Flask remote-control API runs on (on the Pi)
CONTROLLER_PORT = 5000
