#!/usr/bin/env python3
"""
clean_tee.py

Reads from standard input and writes to both standard output and a file.
For the standard output, it passes all bytes exactly as received (including \r).
For the file output, it filters out carriage returns (\r) and the lines they overwrite,
so that progress bar frames (e.g., from tqdm) are not saved to the log file.

Usage:
  python scripts/clean_tee.py <output_file>
"""

import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/clean_tee.py <output_file>")
        sys.exit(1)
        
    log_file = sys.argv[1]
    
    # Open file in append/binary mode
    with open(log_file, "ab") as f:
        buf = bytearray()
        
        while True:
            # Read 1 byte from stdin (unbuffered)
            b = sys.stdin.buffer.read(1)
            
            if not b:
                # EOF: write whatever is left
                if buf:
                    f.write(buf)
                break
                
            # 1. Immediately pass everything to the console
            sys.stdout.buffer.write(b)
            sys.stdout.buffer.flush()
            
            # 2. Process for the log file
            if b == b"\n":
                # End of a real line -> commit the buffer
                buf.extend(b"\n")
                f.write(buf)
                f.flush()
                buf.clear()
            elif b == b"\r":
                # Carriage return -> progress bar frame.
                # We clear the buffer (throwing away the partial line)
                buf.clear()
            else:
                # Accumulate normal characters
                buf.extend(b)

if __name__ == "__main__":
    main()
