class Logger:
    def __init__(self, prefix="ZER"):
        self.prefix = prefix

    def info(self, msg):
        print(f"[{self.prefix}] INFO: {msg}")

    def error(self, msg):
        print(f"[{self.prefix}] ERROR: {msg}")

    def warning(self, msg):
        print(f"[{self.prefix}] WARNING: {msg}")

    def debug(self, msg):
        print(f"[{self.prefix}] DEBUG: {msg}")
