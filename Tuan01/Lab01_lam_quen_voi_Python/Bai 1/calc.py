
import math
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

__all__ = [
    'add', 'sub', 'mul', 'div', 'mod', 'power', 'sqrt',
    'calculate', 'demo',
]


# ----------------------------------------------------------------------
# Các phép toán cơ bản
# ----------------------------------------------------------------------
def add(a, b):
    """Phép cộng: a + b"""
    return a + b


def sub(a, b):
    """Phép trừ: a - b"""
    return a - b


def mul(a, b):
    """Phép nhân: a * b"""
    return a * b


def div(a, b):
    """Phép chia: a / b. Ném ZeroDivisionError nếu b == 0."""
    if b == 0:
        raise ZeroDivisionError('Không thể chia cho 0')
    return a / b


def mod(a, b):
    """Phép chia lấy dư: a % b."""
    if b == 0:
        raise ZeroDivisionError('Không thể lấy dư với 0')
    return a % b


def power(a, b):
    """Phép luỹ thừa: a ** b"""
    return a ** b


def sqrt(a):
    """Căn bậc hai của a (a >= 0)."""
    if a < 0:
        raise ValueError('Không thể lấy căn bậc hai của số âm')
    return math.sqrt(a)


# ----------------------------------------------------------------------
# Tính giá trị một biểu thức dạng chuỗi
# ----------------------------------------------------------------------
_ALLOWED = set('0123456789+-*/%(). ')


def calculate(expression):
    """Tính giá trị biểu thức số học dạng chuỗi, ví dụ '1 + 2 * 3'.

    Chỉ chấp nhận chữ số và các toán tử + - * / % ** ( ) để tránh
    thực thi mã nguy hiểm.
    """
    if not isinstance(expression, str):
        raise TypeError('expression phải là chuỗi')

    if not set(expression) <= _ALLOWED:
        raise ValueError(f'Biểu thức chứa ký tự không hợp lệ: {expression!r}')

    return eval(expression, {'__builtins__': {}}, {})


# ----------------------------------------------------------------------
# Phần trình diễn khi chạy trực tiếp
# ----------------------------------------------------------------------
def demo(a=12, b=5):
    """In bảng kết quả các phép toán trên hai số a, b."""
    print('=' * 46)
    print(f'{"DEMO calc.py":^46s}')
    print('=' * 46)
    print(f'a = {a}, b = {b}\n')

    print(f'{"add(a, b)":<14s} a + b   = {add(a, b)}')
    print(f'{"sub(a, b)":<14s} a - b   = {sub(a, b)}')
    print(f'{"mul(a, b)":<14s} a * b   = {mul(a, b)}')
    print(f'{"div(a, b)":<14s} a / b   = {div(a, b):.4f}')
    print(f'{"mod(a, b)":<14s} a % b   = {mod(a, b)}')
    print(f'{"power(a, b)":<14s} a ** b  = {power(a, b)}')
    print(f'{"sqrt(a)":<14s} sqrt(a) = {sqrt(a):.4f}')

    print('\n-- Tính biểu thức chuỗi --')
    for e in ['1 + 2 * 3', '(4 + 6) / 5', '2 * (3 + 4) - 5']:
        print(f'{e:>16s} = {calculate(e)}')

    print('\n-- Xử lý lỗi --')
    try:
        div(1, 0)
    except ZeroDivisionError as err:
        print(f'div(1, 0) -> ZeroDivisionError: {err}')
    try:
        sqrt(-4)
    except ValueError as err:
        print(f'sqrt(-4)  -> ValueError: {err}')

    print('=' * 46)


if __name__ == '__main__':
    demo()
