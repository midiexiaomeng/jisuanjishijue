import ctypes
import sys

def resume_process(pid):
    """恢复指定PID的进程"""
    PROCESS_SUSPEND_RESUME = 0x0800
    
    # 打开进程
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
    
    if handle == 0:
        print(f"无法打开进程 PID {pid}, 错误代码: {kernel32.GetLastError()}")
        return False
    
    try:
        # 使用NtResumeProcess恢复进程
        ntdll = ctypes.windll.ntdll
        status = ntdll.NtResumeProcess(handle)
        
        if status == 0:
            print(f"成功恢复进程 PID {pid}")
            return True
        else:
            print(f"恢复进程失败, 状态代码: {status}")
            return False
    finally:
        kernel32.CloseHandle(handle)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python resume_process.py <PID>")
        print("示例: python resume_process.py 324076")
        sys.exit(1)
    
    try:
        pid = int(sys.argv[1])
        resume_process(pid)
    except ValueError:
        print("PID必须是整数")
        sys.exit(1)
