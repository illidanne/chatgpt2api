import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from services.image_input_errors import ImageInputError, find_image_input_error

ERROR = ('The user asked to edit or restore a specific existing image, but there is no usable '
         'image target in the conversation. Do not retry image generation. Tell the user to '
         'upload or identify the image before calling image_gen again.')


def conversation(role='tool', text=ERROR):
    return {'mapping': {'tool-result': {'message': {
        'author': {'role': role, 'name': 't2uay3k.sj1i4kz'},
        'status': 'finished_successfully', 'metadata': {},
        'content': {'content_type': 'text', 'parts': [text]},
    }}}}


class ImageInputErrorTests(unittest.TestCase):
    def test_actual_tool_error_without_image_metadata(self):
        self.assertEqual(find_image_input_error(conversation()), ERROR)

    def test_quoted_source_and_ordinary_tool_text_are_not_errors(self):
        for role in ['user', 'assistant']:
            self.assertEqual(find_image_input_error(conversation(role)), '')
        self.assertEqual(find_image_input_error(conversation(text='Generating image...')), '')
        self.assertEqual(find_image_input_error({}), '')

    def test_account_pool_returns_input_error_without_retry(self):
        import __future__
        path = Path(__file__).parents[1] / 'services/protocol/conversation.py'
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and n.name in {'ImageGenerationError', 'stream_image_outputs_with_pool'}]
        accounts = Mock(); accounts.get_available_access_token.return_value = 'test-token'
        namespace = {'ImageInputError': ImageInputError, 'IMAGE_MODELS': ['gpt-image-2'],
                     'account_service': accounts, 'OpenAIBackendAPI': Mock(),
                     'stream_image_outputs': Mock(side_effect=ImageInputError(ERROR, 'test-conversation'))}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec',
                     flags=__future__.annotations.compiler_flag), namespace)
        request = SimpleNamespace(model='gpt-image-2', n=1)
        with self.assertRaises(namespace['ImageGenerationError']) as caught:
            list(namespace['stream_image_outputs_with_pool'](request))
        self.assertEqual(caught.exception.code, 'image_input_missing')
        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(str(caught.exception), ERROR)
        accounts.get_available_access_token.assert_called_once()

    def test_real_polling_method_stops_after_first_tool_error(self):
        # Execute the actual method with a mocked network/clock; no account or dependency setup.
        path = Path(__file__).parents[1] / 'services/openai_backend_api.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OpenAIBackendAPI')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_poll_image_results')
        clock = Mock(); clock.time.return_value = 1
        namespace = {'time': clock, 'logger': Mock(), 'find_image_input_error': find_image_input_error,
                     'ImageInputError': ImageInputError}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        backend = SimpleNamespace(_get_conversation=Mock(return_value=conversation()))
        with self.assertRaises(ImageInputError) as caught:
            namespace['_poll_image_results'](backend, 'conversation-id')
        self.assertEqual(caught.exception.conversation_id, 'conversation-id')
        backend._get_conversation.assert_called_once()
        clock.sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
