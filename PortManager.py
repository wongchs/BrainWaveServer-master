import serial.tools.list_ports


def find_device():
    ports = serial.tools.list_ports.comports()
    serial_port_list = []
    for port in ports:
        serial_port_list.append(port.device)
        print(port)
    return serial_port_list
