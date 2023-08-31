import psutil 

def get_system_resource_consumption():
    return {
        'cpu consumption %': psutil.cpu_percent,
        'memory consumption %': psutil.swap_memory().percent
    }