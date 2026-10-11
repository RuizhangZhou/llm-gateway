import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from llm_gateway import _config


class HostedConfigTests(unittest.TestCase):
    def test_optional_server_path_permission_error_is_ignored(self):
        with patch.object(Path, 'is_file', side_effect=PermissionError('non-root')):
            self.assertFalse(_config._accessible_file(Path('/root/llm-gateway/config/routing.yaml')))

    def test_unreadable_optional_server_env_preserves_injected_credentials(self):
        with patch.object(Path, 'read_text', side_effect=PermissionError('non-root')), \
             patch.dict(os.environ, {'KICONNECT_API_KEY': 'injected-test-token'}):
            _config._load_env()
            self.assertEqual(os.environ['KICONNECT_API_KEY'], 'injected-test-token')

    def test_import_survives_non_root_server_path_probe(self):
        code = '''
from pathlib import Path
original = Path.is_file
def is_file(path):
    if str(path).startswith('/root/'):
        raise PermissionError('non-root runner')
    return original(path)
Path.is_file = is_file
import llm_gateway
print('import succeeded')
'''
        env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('import succeeded', result.stdout)


if __name__ == '__main__':
    unittest.main()
