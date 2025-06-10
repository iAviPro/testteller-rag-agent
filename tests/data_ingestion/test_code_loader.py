import pytest
import asyncio
import os
import shutil # For mocking rmtree
from unittest import mock
from pathlib import Path
import sys # For sys.modules git mock

# Make sure aiofiles is available for mocking
try:
    import aiofiles
    import aiofiles.os as aio_os # Not directly used by CodeLoader but good for completeness
except ImportError:
    # Create dummy aiofiles for environments where it might not be installed
    aiofiles = mock.MagicMock()
    aio_os = mock.MagicMock()

# --- Git Mocking ---
# Mock git before importing CodeLoader, as CodeLoader might import git at module level.
# This setup ensures that when CodeLoader does `import git` or `from git import Repo`,
# it gets our mock.

# Create a mock for the 'git' module
git_module_mock = mock.MagicMock(name="git_module_mock")

# Create a mock for the 'Repo' class within the 'git' module
git_repo_class_mock = mock.MagicMock(name="Repo_class_mock")
git_module_mock.Repo = git_repo_class_mock

# Create a mock for an instance of the 'Repo' class
git_repo_instance_mock = mock.MagicMock(name="Repo_instance_mock")
# Configure Repo class mock to return the instance mock when called (e.g. Repo.clone_from)
# Repo.clone_from is a class method, so it's on Repo class mock directly
git_repo_class_mock.clone_from = mock.MagicMock(name="clone_from_mock")


# Apply the mock to sys.modules BEFORE CodeLoader is imported
# Important: This must happen before `from data_ingestion.code_loader import CodeLoader`
original_sys_modules_git = sys.modules.get('git')
sys.modules['git'] = git_module_mock

# Now import CodeLoader and its dependencies
from data_ingestion.code_loader import CodeLoader
from config import CodeLoaderSettings

# Restore original sys.modules state for 'git' after CodeLoader import,
# so other tests (if any in the same session) are not affected.
if original_sys_modules_git:
    sys.modules['git'] = original_sys_modules_git
else:
    # If 'git' was not in sys.modules, remove our mock
    if 'git' in sys.modules:
        del sys.modules['git']
# --- End Git Mocking ---


import logging # For caplog

# Default extensions from CodeLoaderSettings for comparison
DEFAULT_CODE_EXTENSIONS = CodeLoaderSettings().code_extensions

@pytest.fixture
def code_loader():
    """Fixture to create a CodeLoader instance and reset global state."""
    # Reset global cloned_repos set for each test to ensure isolation
    # This is important because CodeLoader.cloned_repos is a class variable.
    CodeLoader.cloned_repos.clear()
    loader = CodeLoader()
    # Ensure temp_clone_dir_base exists for tests that might try to create subdirs in it,
    # even if actual cloning is mocked.
    # Use a test-specific temp directory.
    loader.temp_clone_dir_base = "./temp_cloned_repos_test"
    os.makedirs(loader.temp_clone_dir_base, exist_ok=True)
    yield loader # Provide the loader instance to the test
    # Teardown: remove the test-specific temp directory
    if os.path.exists(loader.temp_clone_dir_base):
        shutil.rmtree(loader.temp_clone_dir_base)


@pytest.fixture
def mock_aiofiles_open(monkeypatch):
    """Mocks aiofiles.open for reading file content."""
    mock_open = mock.AsyncMock(spec=aiofiles.threadpool.AsyncTextIOWrapper)
    # Default behavior: successfully reads "mock content"
    mock_open.return_value.__aenter__.return_value.read.return_value = "mock content"
    monkeypatch.setattr("aiofiles.open", mock_open)
    return mock_open

# --- Test Initialization ---
def test_codeloader_init(code_loader):
    assert code_loader.code_extensions == DEFAULT_CODE_EXTENSIONS
    assert code_loader.temp_clone_dir_base == "./temp_cloned_repos_test"
    assert CodeLoader.cloned_repos == set()

# --- Test _is_supported_file ---
@pytest.mark.parametrize("filepath, expected", [
    ("test.py", True),
    ("path/to/file.js", True),
    ("archive.zip", False),
    ("nodotextension", False),
    ("image.JPG", False),
    ("path/to/.env", ".env" in DEFAULT_CODE_EXTENSIONS), # Check based on actual defaults
    ("file.with.dots.py", True),
    ("file.with.dots.txt", True), # .txt is often not in CODE_EXTENSIONS by default
    ("unsupported.with.dots.exe", False),
])
def test_is_supported_file(code_loader, filepath, expected):
    if filepath == "file.with.dots.txt": # .txt might not be a default code extension
        expected = ".txt" in code_loader.code_extensions
    assert code_loader._is_supported_file(filepath) == expected

# --- Test _read_code_files_from_path ---
@pytest.mark.asyncio
async def test_read_code_files_from_path_mixed_files(code_loader, mock_aiofiles_open, caplog):
    base_path_str = "/testrepo"
    base_path = Path(base_path_str)

    # Define mock Path objects that rglob would return
    # Each needs is_file(), __str__()
    file_mocks_data = [
        {"path_str": "/testrepo/file1.py", "is_file": True, "content": "python content"},
        {"path_str": "/testrepo/module/file2.js", "is_file": True, "content": "javascript content"},
        {"path_str": "/testrepo/data.unsupported", "is_file": True}, # No content needed, will be skipped
        {"path_str": "/testrepo/docs", "is_file": False}, # A directory
        {"path_str": "/testrepo/empty.py", "is_file": True, "content": ""},
        {"path_str": "/testrepo/error.py", "is_file": True, "error": IOError("mock read error")},
    ]

    rglob_results = []
    for data in file_mocks_data:
        p_mock = mock.MagicMock(spec=Path)
        p_mock.is_file.return_value = data["is_file"]
        p_mock.__str__.return_value = data["path_str"]
        p_mock.name = os.path.basename(data["path_str"]) # For logging unsupported
        rglob_results.append(p_mock)

    # Configure mock_aiofiles_open for specific files
    async def custom_aio_open_side_effect(filepath_str, mode, encoding, errors):
        mock_file_wrapper = mock.AsyncMock(spec=aiofiles.threadpool.AsyncTextIOWrapper)
        for data in file_mocks_data:
            if data["path_str"] == filepath_str:
                if "error" in data:
                    mock_file_wrapper.__aenter__.return_value.read.side_effect = data["error"]
                else:
                    mock_file_wrapper.__aenter__.return_value.read.return_value = data.get("content", "default mock content")
                break
        return mock_file_wrapper
    mock_aiofiles_open.side_effect = custom_aio_open_side_effect

    caplog.set_level(logging.INFO) # To capture skip/error logs

    results = []
    # Mock Path.rglob for the specific base_path instance
    with mock.patch.object(Path, 'rglob', return_value=rglob_results) as mock_rglob_method:
        # We need to ensure that when Path(base_path_str) is called inside the method,
        # its rglob method is this mock_rglob_method.
        # This requires mocking the Path constructor or carefully patching the instance.

        # Simpler: Assume _read_code_files_from_path is called with a Path instance.
        # Let's make a mock Path instance for base_path and set its rglob.
        mock_base_path_instance = mock.MagicMock(spec=Path)
        mock_base_path_instance.rglob.return_value = rglob_results
        mock_base_path_instance.__str__.return_value = base_path_str

        async for source_id, content in code_loader._read_code_files_from_path(mock_base_path_instance, "local_test"):
            results.append((source_id, content))

        mock_base_path_instance.rglob.assert_called_once_with("*")

    assert len(results) == 2 # file1.py, file2.js
    expected_results = [
        ("local_test:/testrepo/file1.py", "python content"),
        ("local_test:/testrepo/module/file2.js", "javascript content"),
    ]
    results.sort(key=lambda x: x[0])
    expected_results.sort(key=lambda x: x[0])
    assert results == expected_results

    assert "Skipping unsupported file type: /testrepo/data.unsupported" in caplog.text
    assert "Skipping empty file: /testrepo/empty.py" in caplog.text
    assert "Error reading file /testrepo/error.py: mock read error" in caplog.text


# --- Test load_code_from_local_folder ---
@pytest.mark.asyncio
async def test_load_code_from_local_folder_valid_path(code_loader, monkeypatch, caplog):
    local_path_str = "/local/folder"
    resolved_path_str = "/resolved/local/folder"

    # Mock for Path(local_path_str)
    mock_path_obj = mock.MagicMock(spec=Path)
    mock_path_obj.exists.return_value = True
    mock_path_obj.is_dir.return_value = True
    mock_path_obj.resolve.return_value = Path(resolved_path_str) # Return a new Path obj for resolved

    # Mock Path constructor to return our mock_path_obj when called with local_path_str
    # And a different one for the resolved path.
    def path_constructor(p):
        if str(p) == local_path_str:
            return mock_path_obj
        elif str(p) == resolved_path_str: # For Path(resolved_path_str) inside helper
            resolved_mock = mock.MagicMock(spec=Path)
            resolved_mock.__str__.return_value = resolved_path_str
            # _read_code_files_from_path will call rglob on this
            resolved_mock.rglob.return_value = [] # Default to no files for simplicity here
            return resolved_mock
        return Path(p) # Fallback to actual Path for other cases (e.g. inside helper)

    monkeypatch.setattr("data_ingestion.code_loader.Path", path_constructor)

    # Mock the helper method _read_code_files_from_path
    async def mock_read_generator(path_obj, source_type):
        # path_obj here should be the resolved path
        assert str(path_obj) == resolved_path_str
        assert source_type == "local"
        yield (f"local:{resolved_path_str}/file1.py", "content1")

    with mock.patch.object(CodeLoader, '_read_code_files_from_path', side_effect=mock_read_generator) as mock_read_helper:
        results = await code_loader.load_code_from_local_folder(local_path_str)

    assert len(results) == 1
    assert results[0] == (f"local:{resolved_path_str}/file1.py", "content1")
    mock_path_obj.exists.assert_called_once()
    mock_path_obj.is_dir.assert_called_once()
    mock_path_obj.resolve.assert_called_once()
    mock_read_helper.assert_called_once()
    # Check the first argument of the call to mock_read_helper
    assert str(mock_read_helper.call_args[0][0]) == resolved_path_str


@pytest.mark.asyncio
async def test_load_code_from_local_folder_invalid_paths(code_loader, monkeypatch, caplog):
    caplog.set_level(logging.ERROR)

    # Test non-existent path
    mock_path_nonexist = mock.MagicMock(spec=Path)
    mock_path_nonexist.exists.return_value = False
    monkeypatch.setattr("data_ingestion.code_loader.Path", lambda p: mock_path_nonexist)
    results = await code_loader.load_code_from_local_folder("/nonexistent")
    assert len(results) == 0
    assert "Local code path /nonexistent does not exist." in caplog.text

    # Test path is not a directory
    mock_path_notdir = mock.MagicMock(spec=Path)
    mock_path_notdir.exists.return_value = True
    mock_path_notdir.is_dir.return_value = False
    monkeypatch.setattr("data_ingestion.code_loader.Path", lambda p: mock_path_notdir)
    results = await code_loader.load_code_from_local_folder("/not/a/dir")
    assert len(results) == 0
    assert "Local code path /not/a/dir is not a directory." in caplog.text

# --- Test load_code_from_repo ---
@pytest.mark.asyncio
async def test_load_code_from_repo_success(code_loader, monkeypatch, caplog):
    repo_url = "httpsgitscheme://github.com/user/repo.git" # Use a non-http scheme to avoid actual net call
    # This relies on git_module_mock.Repo.clone_from being available (setup in imports)

    # _repo_url_to_dirname needs to be consistent
    # Replicate its logic simply for the test:
    expected_dirname = "repo.git" # Simplified from actual helper
    expected_temp_path_str = os.path.join(code_loader.temp_clone_dir_base, expected_dirname)

    # Mock Path object behavior for the expected temporary path
    mock_cloned_path_obj = mock.MagicMock(spec=Path)
    mock_cloned_path_obj.__str__.return_value = expected_temp_path_str
    mock_cloned_path_obj.exists.return_value = True # After "cloning"
    mock_cloned_path_obj.resolve.return_value = mock_cloned_path_obj # Resolve returns self

    # Mock Path constructor:
    # 1. For base temp dir (CodeLoader.temp_clone_dir_base)
    # 2. For the target clone path (expected_temp_path_str)
    def path_constructor_for_repo(p_str):
        path_str_resolved = str(Path(p_str).resolve()) # Resolve to normalize
        if path_str_resolved == str(Path(code_loader.temp_clone_dir_base).resolve()):
            # Base temp directory
            base_mock = mock.MagicMock(spec=Path)
            base_mock.__str__.return_value = path_str_resolved
            base_mock.exists.return_value = True # Assume base temp dir exists
            base_mock.is_dir.return_value = True
            base_mock.__truediv__ = lambda s, k: Path(os.path.join(str(s), k)) # for / operator
            return base_mock
        elif path_str_resolved == str(Path(expected_temp_path_str).resolve()):
            return mock_cloned_path_obj
        # Fallback for other Path uses (e.g. inside _read_code_files_from_path if not fully mocked)
        real_path = Path(p_str)
        # print(f"Path constructor called with '{p_str}', returning real Path object for some cases.")
        return real_path

    monkeypatch.setattr("data_ingestion.code_loader.Path", path_constructor_for_repo)
    monkeypatch.setattr("os.makedirs", mock.MagicMock()) # Mock makedirs

    # Mock _read_code_files_from_path
    async def mock_read_generator(path_obj, source_type):
        assert str(path_obj) == expected_temp_path_str # Should be called with the cloned path
        assert source_type == repo_url
        yield (f"{repo_url}:{str(path_obj)}/file.py", "cloned content")

    with mock.patch.object(CodeLoader, '_read_code_files_from_path', side_effect=mock_read_generator) as mock_read_helper:
        # Ensure clone_from is reset and available on the class mock
        git_module_mock.Repo.clone_from.reset_mock()
        git_module_mock.Repo.clone_from.side_effect = None # Clear any previous side effects

        results = await code_loader.load_code_from_repo(repo_url, cleanup_after=False)

    assert len(results) == 1
    assert results[0] == (f"{repo_url}:{expected_temp_path_str}/file.py", "cloned content")

    # Check clone_from was called with correct URL and path
    # The path would be resolved.
    resolved_expected_temp_path_str = str(Path(expected_temp_path_str).resolve())
    git_module_mock.Repo.clone_from.assert_called_once_with(repo_url, resolved_expected_temp_path_str)

    assert resolved_expected_temp_path_str in CodeLoader.cloned_repos
    mock_read_helper.assert_called_once()
    # Ensure the first arg to mock_read_helper is a Path obj that stringifies to resolved_expected_temp_path_str
    assert str(mock_read_helper.call_args[0][0]) == resolved_expected_temp_path_str


@pytest.mark.asyncio
async def test_load_code_from_repo_clone_fails(code_loader, monkeypatch, caplog):
    repo_url = "httpsgitscheme://github.com/user/repo.git"
    # Configure clone_from mock on the git_module_mock.Repo (which is git_repo_class_mock)
    git_module_mock.Repo.clone_from.side_effect = Exception("Clone failed miserably")
    git_module_mock.Repo.clone_from.reset_mock() # Ensure it's clean for this test if needed
    git_module_mock.Repo.clone_from.side_effect = Exception("Clone failed miserably")


    caplog.set_level(logging.ERROR)
    monkeypatch.setattr("os.makedirs", mock.MagicMock()) # Mock makedirs

    results = await code_loader.load_code_from_repo(repo_url)

    assert len(results) == 0
    assert f"Failed to clone repository {repo_url}: Clone failed miserably" in caplog.text
    assert not CodeLoader.cloned_repos

# --- Test cleanup_repo and cleanup_all_repos ---
@pytest.mark.asyncio
async def test_cleanup_repo(code_loader, monkeypatch):
    repo_url = "httpsgitscheme://github.com/user/repo.git"
    dirname = code_loader._repo_url_to_dirname(repo_url) # Use the actual helper
    temp_repo_path_str = str(Path(code_loader.temp_clone_dir_base).resolve() / dirname)

    CodeLoader.cloned_repos.add(temp_repo_path_str) # Manually add to simulate it was cloned

    mock_path_obj = mock.MagicMock(spec=Path)
    mock_path_obj.__str__.return_value = temp_repo_path_str
    mock_path_obj.exists.return_value = True
    mock_path_obj.is_dir.return_value = True

    monkeypatch.setattr("data_ingestion.code_loader.Path", lambda p: mock_path_obj if str(Path(p).resolve()) == temp_repo_path_str else Path(p))

    with mock.patch("shutil.rmtree") as mock_rmtree:
        await code_loader.cleanup_repo(repo_url)
        mock_rmtree.assert_called_once_with(mock_path_obj) # shutil.rmtree is called with Path obj
        assert temp_repo_path_str not in CodeLoader.cloned_repos


@pytest.mark.asyncio
async def test_cleanup_all_repos(code_loader, monkeypatch):
    path1_str_resolved = str(Path(code_loader.temp_clone_dir_base).resolve() / "repo1")
    path2_str_resolved = str(Path(code_loader.temp_clone_dir_base).resolve() / "repo2")
    CodeLoader.cloned_repos.add(path1_str_resolved)
    CodeLoader.cloned_repos.add(path2_str_resolved)

    # Mock Path constructor to return paths that exist and are directories
    # for the paths we added to cloned_repos.
    path_mocks = {}
    def path_constructor_side_effect(p_str_arg):
        p_str = str(Path(p_str_arg).resolve()) # Normalize
        if p_str not in path_mocks:
            instance = mock.MagicMock(spec=Path)
            instance.__str__.return_value = p_str # Store resolved path string
            instance.exists.return_value = p_str in CodeLoader.cloned_repos
            instance.is_dir.return_value = p_str in CodeLoader.cloned_repos
            path_mocks[p_str] = instance
        return path_mocks[p_str]

    monkeypatch.setattr("data_ingestion.code_loader.Path", path_constructor_side_effect)

    with mock.patch("shutil.rmtree") as mock_rmtree:
        await code_loader.cleanup_all_repos()

        assert mock_rmtree.call_count == 2
        # shutil.rmtree is called with Path objects.
        # Check that the string representations of the Path objects match.
        called_paths_str = sorted([str(call_arg[0][0]) for call_arg in mock_rmtree.call_args_list])
        assert called_paths_str == sorted([path1_str_resolved, path2_str_resolved])
        assert not CodeLoader.cloned_repos
```
