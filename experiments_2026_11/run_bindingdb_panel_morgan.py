# -*- coding: utf-8 -*-
"""
兼容入口：等价于
    python run_panel_morgan.py --dataset bindingdb ...

保留该文件是为了让此前记录的命令仍然可用。新命令请优先使用 run_panel_morgan.py。
"""
import sys

from run_panel_morgan import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--dataset", "bindingdb"] + sys.argv[1:]
    main()
