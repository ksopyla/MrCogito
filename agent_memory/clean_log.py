import sys

def clean_log(input_path, output_path):
    with open(input_path, 'rb') as f:
        data = f.read()

    lines = []
    buf = bytearray()
    for b in data:
        if b == ord('\n'):
            buf.append(b)
            lines.append(bytes(buf))
            buf = bytearray()
        elif b == ord('\r'):
            buf = bytearray()
        else:
            buf.append(b)
    if buf:
        lines.append(bytes(buf))

    with open(output_path, 'wb') as f:
        for line in lines:
            f.write(line)

if __name__ == '__main__':
    clean_log('agent_memory/shell_diffusion_20260221_195541.log', 'agent_memory/cleaned_log.txt')
    print("Log cleaned.")