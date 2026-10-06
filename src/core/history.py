# -*- coding: utf-8 -*-
"""撤销 / 重做。

因为核心骨架阶段没有像素级编辑，快照用的是"浅克隆"（图层对象新建、
numpy 数组共享），所以一步撤销的内存开销几乎可以忽略。
"""

from __future__ import annotations


class History:
    def __init__(self, limit=40):
        self.stack = []
        self.index = -1
        self.limit = limit

    def reset(self, doc):
        self.stack = [doc.clone()]
        self.index = 0

    def commit(self, doc):
        """记录一次变更后（或变更前）的状态。"""
        # 丢弃当前位置之后的 redo 分支
        if self.index < len(self.stack) - 1:
            del self.stack[self.index + 1:]
        self.stack.append(doc.clone())
        if len(self.stack) > self.limit:
            del self.stack[0]
        self.index = len(self.stack) - 1

    def can_undo(self):
        return self.index > 0

    def can_redo(self):
        return self.index < len(self.stack) - 1

    def undo(self):
        if not self.can_undo():
            return None
        self.index -= 1
        return self.stack[self.index].clone()

    def redo(self):
        if not self.can_redo():
            return None
        self.index += 1
        return self.stack[self.index].clone()
