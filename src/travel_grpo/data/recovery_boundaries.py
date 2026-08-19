"""Compatibility facade for :mod:`travel_grpo.data.recovery.boundaries`."""

from travel_grpo.data.recovery import boundaries as _canonical
from travel_grpo.data.recovery.boundaries import *  # noqa: F401,F403


# [项目注释] 功能：`__getattr__`：实现该模块在当前调用链中的局部业务逻辑，并维护相关状态不变量。 主要协作调用：getattr。
# [项目注释] 输入：`name`: str。
# [项目注释] 输出：返回运行时计算结果；没有显式返回值的分支返回 `None`。
def __getattr__(name: str):
    return getattr(_canonical, name)
