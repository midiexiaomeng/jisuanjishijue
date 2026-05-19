import ctypes
import sys

def suspend_process(pid):
    """暂停指定PID的进程"""
    PROCESS_SUSPEND_RESUME = 0x0800
    PROCESS_TERMINATE = 0x0001
    
    # 打开进程
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
    
    if handle == 0:
        print(f"无法打开进程 PID {pid}, 错误代码: {kernel32.GetLastError()}")
        return False
    
    try:
        # 尝试使用NtSuspendProcess（需要从ntdll导入）
        ntdll = ctypes.windll.ntdll
        status = ntdll.NtSuspendProcess(handle)
        
        if status == 0:
            print(f"成功暂停进程 PID {pid}")
            return True
        else:
            print(f"暂停进程失败, 状态代码: {status}")
            return False
    finally:
        kernel32.CloseHandle(handle)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python pause_process.py <PID>")
        sys.exit(1)
    
    try:
        pid = int(sys.argv[1])
        suspend_process(pid)
    except ValueError:
        print("PID必须是整数")
        sys.exit(1)
