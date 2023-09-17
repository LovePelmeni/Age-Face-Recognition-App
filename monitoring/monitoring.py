import psutil 

def get_system_resource_consumption():
    return {
        'cpu consumption %': round(psutil.cpu_percent(interval=1) / 100, 2),
        'memory consumption %': psutil.virtual_memory()
    }