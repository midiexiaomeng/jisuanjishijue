import ctypes
import sys
import subprocess

def get_python_processes():
    """获取所有Python进程信息"""
    try:
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV', '/NH'],
            capture_output=True,
            text=True,
            encoding='gbk'
        )
        processes = []
        for line in result.stdout.strip().split('\n'):
            if line and 'python.exe' in line:
                parts = line.strip('"').split('","')
                if len(parts) >= 5:
                    pid = int(parts[1])
                    memory = parts[4].replace(' K', '').replace(',', '')
                    memory_kb = int(memory) if memory.isdigit() else 0
                    processes.append({
                        'name': parts[0],
                        'pid': pid,
                        'memory_kb': memory_kb,
                        'memory_mb': memory_kb / 1024
                    })
        return processes
    except Exception as e:
        print(f"获取进程列表失败: {e}")
        return []

def suspend_process(pid):
    """暂停指定PID的进程"""
    PROCESS_SUSPEND_RESUME = 0x0800
    
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
    
    if handle == 0:
        print(f"无法打开进程 PID {pid}, 错误代码: {kernel32.GetLastError()}")
        return False
    
    try:
        ntdll = ctypes.windll.ntdll
        status = ntdll.NtSuspendProcess(handle)
        
        if status == 0:
            print(f"✓ 成功暂停进程 PID {pid}")
            return True
        else:
            print(f"✗ 暂停进程失败, 状态代码: {status}")
            return False
    finally:
        kernel32.CloseHandle(handle)

def resume_process(pid):
    """恢复指定PID的进程"""
    PROCESS_SUSPEND_RESUME = 0x0800
    
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
    
    if handle == 0:
        print(f"无法打开进程 PID {pid}, 错误代码: {kernel32.GetLastError()}")
        return False
    
    try:
        ntdll = ctypes.windll.ntdll
        status = ntdll.NtResumeProcess(handle)
        
        if status == 0:
            print(f"✓ 成功恢复进程 PID {pid}")
            return True
        else:
            print(f"✗ 恢复进程失败, 状态代码: {status}")
            return False
    finally:
        kernel32.CloseHandle(handle)

def main():
    """主函数"""
    print("=" * 50)
    print("Python进程管理器")
    print("=" * 50)
    
    processes = get_python_processes()
    
    if not processes:
        print("未找到Python进程")
        return
    
    print(f"\n找到 {len(processes)} 个Python进程:")
    print("-" * 60)
    print(f"{'序号':<5} {'PID':<10} {'内存(MB)':<12} {'进程名'}")
    print("-" * 60)
    
    for i, proc in enumerate(processes, 1):
        print(f"{i:<5} {proc['pid']:<10} {proc['memory_mb']:<12.1f} {proc['name']}")
    
    print("\n操作选项:")
    print("1. 暂停进程")
    print("2. 恢复进程")
    print("3. 退出")
    
    choice = input("\n请选择操作 (1-3): ").strip()
    
    if choice == '1':
        pid_input = input("请输入要暂停的进程PID: ").strip()
        try:
            pid = int(pid_input)
            suspend_process(pid)
        except ValueError:
            print("错误: PID必须是整数")
    elif choice == '2':
        pid_input = input("请输入要恢复的进程PID: ").strip()
        try:
            pid = int(pid_input)
            resume_process(pid)
        except ValueError:
            print("错误: PID必须是整数")
    elif choice == '3':
        print("退出进程管理器")
    else:
        print("无效的选择")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        # 命令行模式
        arg = sys.argv[1]
        if arg == 'list':
            processes = get_python_processes()
            for proc in processes:
                print(f"PID: {proc['pid']}, 内存: {proc['memory_mb']:.1f}MB")
        elif arg.startswith('suspend:'):
            pid = int(arg.split(':')[1])
            suspend_process(pid)
        elif arg.startswith('resume:'):
            pid = int(arg.split(':')[1])
            resume_process(pid)
        else:
            print("用法:")
            print("  python process_manager.py list          # 列出所有Python进程")
            print("  python process_manager.py suspend:PID   # 暂停指定PID的进程")
            print("  python process_manager.py resume:PID    # 恢复指定PID的进程")
            print("  python process_manager.py              # 进入交互模式")
    else:
        main()
