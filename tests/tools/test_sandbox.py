from labi.tools.python.sandbox import extract_code, execute_code


def test_extract_code_pulls_fenced_python_block():
    text = "Here's the code:\n```python\nprint('hi')\n```\nDone."
    assert extract_code(text) == "print('hi')"


def test_extract_code_falls_back_to_stripped_text_without_fence():
    assert extract_code("  print('hi')  ") == "print('hi')"


def test_execute_code_runs_successfully_and_captures_stdout():
    stdout, stderr, exit_code = execute_code("print('hello world')")
    assert exit_code == 0
    assert "hello world" in stdout
    assert stderr == ""


def test_execute_code_captures_stderr_and_nonzero_exit_on_exception():
    stdout, stderr, exit_code = execute_code("raise ValueError('boom')")
    assert exit_code != 0
    assert "ValueError" in stderr
    assert "boom" in stderr


def test_execute_code_extracts_fenced_code_before_running():
    stdout, stderr, exit_code = execute_code("```python\nprint('fenced')\n```")
    assert exit_code == 0
    assert "fenced" in stdout


def test_execute_code_blocks_dangerous_code_before_running():
    # os.system / subprocess-spawning patterns are exactly what
    # tools/python/security.py's static validator exists to catch --
    # exit_code -2 means "blocked before execution", not "ran and failed".
    stdout, stderr, exit_code = execute_code("import os\nos.system('rm -rf /tmp/whatever')")
    assert exit_code == -2
    assert "Blocked before execution" in stderr


def test_execute_code_times_out_on_infinite_loop():
    # Two independent limits guard against a hang here: subprocess.run's
    # wall-clock timeout (-> exit_code -1, "Execution timed out") and the
    # CPU rlimit from make_preexec_fn (-> killed by SIGXCPU, exit_code
    # -24). Both are set to the same timeout value, so which one actually
    # fires first is a race, not a guarantee -- the real contract under
    # test is "a CPU-bound infinite loop cannot run forever", not "it
    # always returns this exact code via this exact mechanism".
    stdout, stderr, exit_code = execute_code("while True:\n    pass", timeout=1)
    assert exit_code != 0
    assert exit_code < 0  # killed abnormally (timeout or signal), not a normal Python exit


def test_execute_code_truncates_oversized_output():
    stdout, stderr, exit_code = execute_code("print('x' * 10000)", max_output_bytes=100)
    assert len(stdout) <= 150  # allow for a small truncation marker
