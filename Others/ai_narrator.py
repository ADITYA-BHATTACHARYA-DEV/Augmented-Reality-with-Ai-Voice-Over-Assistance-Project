import ollama
import threading

class AINarrator:
    def __init__(self, model="llama3"):
        self.model   = model
        self.reply   = ""
        self.busy    = False

    def ask(self, prompt: str):
        """Non-blocking: fire query in background thread."""
        if self.busy:
            return
        self.busy = True
        threading.Thread(target=self._query, args=(prompt,),
                         daemon=True).start()

    def _query(self, prompt):
        try:
            res = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": ("You are a helpful AR car guide. "
                                 "Give concise answers about car parts, "
                                 "specs, or features in 1-2 sentences.")},
                    {"role": "user", "content": prompt}
                ]
            )
            self.reply = res["message"]["content"]
        except Exception as e:
            self.reply = f"[AI error: {e}]"
        finally:
            self.busy = False

    def get_reply(self):
        return self.reply