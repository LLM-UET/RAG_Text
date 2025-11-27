import time

LOG_FILENAME = "rag.log"

def log(message: str):
    lg = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " " + message + "\n"
    with open(LOG_FILENAME, "a") as f:
        f.write(lg)
    print(lg, end='')

def log_info(message: str):
    log("[INFO] " + message)

def log_error(message: str):
    log("[ERROR] " + message)

def log_warning(message: str):
    log("[WARNING] " + message)

def log_debug(message: str):
    log("[DEBUG] " + message)
