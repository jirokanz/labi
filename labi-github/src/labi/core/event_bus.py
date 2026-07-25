class SyncEventBus:
    def __init__(self):
        self.handlers = {}
    def subscribe(self, event, handler):
        self.handlers.setdefault(event, []).append(handler)
    def emit(self, event, data):
        for h in self.handlers.get(event, []):
            h(data)
