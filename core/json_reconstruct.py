import json

class JsonReconstruct:
    def __init__(self):
        self.buffer = ""

    def process_part(self, chunk, callback):
        if not self.buffer:
            try:
                decoded = json.loads(chunk)
                callback(decoded)
                return
            except json.JSONDecodeError:
                print(f"!!! failed decode: {chunk}")
                self.buffer += chunk
                return

        self.buffer += chunk
        try:
            decoded = json.loads(self.buffer)
            callback(decoded)
            self.buffer = ""
        except json.JSONDecodeError:
            print(f"!!! failed decode: {self.buffer}")
            pass

    def finalize(self, callback):
        if self.buffer:
            try:
                callback(json.loads(self.buffer))
            except json.JSONDecodeError:
                pass
            self.buffer = ""

    def get_buffer(self):
        return self.buffer

