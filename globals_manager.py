# The static global variable
_GLOBAL_DATA = {}

def set_global(key, value):
    """Set a value in the global storage"""
    global _GLOBAL_DATA
    _GLOBAL_DATA[key] = value

def get_global(key, default=None):
    """Get a value from the global storage"""
    global _GLOBAL_DATA
    return _GLOBAL_DATA.get(key, default)

def clear_global(key=None):
    """Clear either a specific key or all globals"""
    global _GLOBAL_DATA
    if key is None:
        _GLOBAL_DATA.clear()
    else:
        _GLOBAL_DATA.pop(key, None)