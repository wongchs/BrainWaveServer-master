def start_socket_server(board, host='0.0.0.0', port=5000):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((host, port))
    except socket.error as e:
        print(f"Socket binding failed: {e}")
        return

    server_socket.listen(1)

    print(f"Waiting for connection on {host}:{port}...")
    try:
        client_socket, client_address = server_socket.accept()
        print(f"Accepted connection from {client_address}")

        while True:
            data = board.get_current_board_data(20)  # Get latest data points
            if data is not None and data.size > 0:
                json_data = json.dumps(data.tolist())
                client_socket.send(json_data.encode())
            time.sleep(0.1)  # Adjust the delay as needed
    except Exception as e:
        print(f"Error in socket server: {e}")
    finally:
        client_socket.close()
        server_socket.close()