import datetime
import subprocess

class Toolbox:
    def __init__(self):
        self.available_tools = {
            "get_time": self.get_time,
            "calculate": self.calculate,
            "system_status": self.system_status
        }

    def get_time(self, args=None):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def calculate(self, expression):
        # SAFETY: Using eval is dangerous. In production, use a parser.
        # For prototype, we restrict chars.
        allowed = set("0123456789+-*/(). ")
        if set(expression).issubset(allowed):
            try:
                return str(eval(expression))
            except:
                return "Error in calculation."
        return "Invalid characters in expression."

    def system_status(self, args=None):
        # Works on Linux/Codespaces
        try:
            uptime = subprocess.check_output("uptime", shell=True).decode()
            return f"System Uptime: {uptime}"
        except:
            return "Could not retrieve system status."

    def execute(self, tool_name, args):
        if tool_name in self.available_tools:
            return self.available_tools[tool_name](args)
        return "Tool not found."