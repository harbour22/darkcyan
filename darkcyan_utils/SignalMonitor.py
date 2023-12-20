import signal

class SignalMonitor:
  exit_now = False
  exited_with_error = False
  def __init__(self):
      print("Signal Monitor created")
      signal.signal(signal.SIGINT, self.exit_gracefully)
      signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self, *args):
      print("Signal Monitor to Exit Now")
      self.exit_now = True

  def exit_with_error(self, *args):
      print("Signal Monitor to Exit Now")
      self.exit_now = True
      self.exited_with_error = True    

  def in_error_state(self, *args):
    return self.exited_with_error