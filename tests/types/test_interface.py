import pytest as pyt


class test_Interface:
    def test(self):
        try:
            import colorama

            pre, suf = colorama.Fore.RED, colorama.Style.RESET_ALL
        except ImportError:
            pre = suf = ""
        pyt.skip(pre + (" ".join(("__NOT_IMPLEMENTED__" * 2,) * 3)) + suf)
