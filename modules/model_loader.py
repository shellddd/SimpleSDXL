import os
import anyio
import httpx
from tqdm import tqdm
import threading
import queue
import json
import ast
import shared
from urllib.parse import urlparse
from typing import Optional
import logging
from enhanced.logger import format_name
logger = logging.getLogger(format_name(__name__))


async def download_file_with_progress(url: str, file_path: str):
    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            if 'HF_MIRROR' in os.environ:
                url = str.replace(url, "huggingface.co", os.environ["HF_MIRROR"].rstrip('/'), 1)
            model_dir = os.path.dirname(file_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("Content-Length", 0))

                with tqdm(
                    total=total_size, unit="iB", unit_scale=True, desc=''
                ) as progress_bar:
                    partial_file_path = file_path + ".partial"
                    with open(partial_file_path, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
                            progress_bar.update(len(chunk))
            os.rename(partial_file_path, file_path)
            shared.modelsinfo.refresh_file('add', file_path, url)
        except httpx.HTTPStatusError as e:
            logger.error(f"下载失败: {e}")
            logger.error(f"请求 URL: {e.request.url}")
            logger.error(f"重定向 URL: {e.response.headers.get('Location')}")
            raise
        except Exception as e:
            logger.error(f"下载过程中发生错误: {e}")
            raise

async def download_multiple_files(task_list):
    for task_type, task_params in task_list:
        if task_type == 'file':
            await download_file_with_progress(task_params['url'], task_params['path_file'])
        elif task_type == 'diffusers':
            await download_diffusers_model(task_params['cata'], task_params['model_name'], task_params['num'], task_params['url'])


def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
        async_task: bool = False,
) -> str:
    global download_queue

    """
    Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    if 'HF_MIRROR' in os.environ:
        url = str.replace(url, "huggingface.co", os.environ["HF_MIRROR"].rstrip('/'), 1)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        logger.info(f'Downloading: "{url}" to {cached_file}')
        logger.info(f'正在下载模型文件: "{url}"。如果速度慢，可终止运行，自行用工具下载后保存到: {cached_file}，然后重启应用。\n')
        
        def _download_task():
            anyio.run(download_file_with_progress, url, cached_file)

        if async_task:
            #download_queue.put(lambda: download_file_with_progress(url, cached_file))  
            thread = threading.Thread(target=_download_task)
            thread.start()
        else:
            from torch.hub import download_url_to_file
            download_url_to_file(url, cached_file, progress=progress)
            shared.modelsinfo.refresh_file('add', cached_file, url)
    return cached_file


presets_model_list = {}
presets_mtime = {}

def refresh_model_list(presets, user_did=None):
    from enhanced.simpleai import get_path_in_user_dir
    global presets_model_list, presets_mtime

    path_preset = os.path.abspath(f'./presets/')
    if user_did:
        user_path_preset = get_path_in_user_dir('presets', user_did)
    if len(presets)>0:
        for preset in presets:
            if preset.endswith('.'):
                if user_did is None:
                    continue
                preset_file = os.path.join(user_path_preset, f'{preset}json')
                preset = f'{preset}{user_did[:7]}'
            else:
                preset_file = os.path.join(path_preset, f'{preset}.json')
            try:
                mtime = os.path.getmtime(preset_file)
                if preset not in presets_mtime:
                    presets_mtime[preset] = 0
                if mtime>presets_mtime[preset]:
                    presets_mtime[preset] = mtime
                    with open(preset_file, "r", encoding="utf-8") as json_file:
                        config_preset = json.load(json_file)
                    if 'model_list' in config_preset:
                        model_list = config_preset['model_list']
                        model_list = [tuple(p.split(',')) for p in model_list]
                        model_list = [(cata.strip(), path_file.strip(), int(size), hash10.strip(), url.strip()) for (cata, path_file, size, hash10, url) in model_list]
                        presets_model_list[preset] = model_list
            except Exception as e:
                logger.info(f'load preset file failed: {preset_file}')
                continue
    return
            

def check_models_exists(preset, user_did=None):
    from modules.config import path_models_root
    global presets_model_list

    if preset.endswith('.'):
        if user_did is None:
            return False
        preset = f'{preset}{user_did[:7]}'
    model_list = [] if preset not in presets_model_list else presets_model_list[preset]
    if len(model_list)>0:
        for cata, path_file, size, hash10, url in model_list:
            if path_file[:1]=='[' and path_file[-1:]==']':
                path_file = [path_file[1:-1]]
                result = shared.modelsinfo.get_model_names(cata, path_file, casesensitive=True)
                if result is None or len(result)<size:
                    #logger.info(f'[ModelInfos] Missing model dir in preset({preset}): {cata}, filter={path_file}, len={size}\nresult={result}')
                    return False
            else:
                file_path = shared.modelsinfo.get_model_filepath(cata, path_file)
                if file_path is None or file_path == '' or not os.path.exists(file_path) or size != os.path.getsize(file_path):
                    logger.info(f'Missing model file in preset({preset}): {cata}, {path_file}')
                    return False
        return True
    return False

default_download_url_prefix = 'https://huggingface.co/metercai/SimpleSDXL2/resolve/main/SimpleModels'
def download_model_files(preset, user_did=None, async_task=False):
    from modules.config import path_models_root, model_cata_map
    global presets_model_list, default_download_url_prefix, download_queue
    
    from others.model_async_downloader import ready_to_download_url, download_it_from_ready_list
    
    if preset.endswith('.'):
        if user_did is None:
            return False
        preset = f'{preset}{user_did[:7]}'
    model_list = [] if preset not in presets_model_list else presets_model_list[preset]
    if len(model_list)>0:
        download_task_list = []
        for cata, path_file, size, hash10, url in model_list:
            download_task = ('', {})
            if path_file[:1]=='[' and path_file[-1:]==']':
                if url:
                    parts = urlparse(url)
                    file_name = os.path.basename(parts.path)
                else:
                    continue
            else:
                file_name = path_file.replace('\\', '/').replace(os.sep, '/')
            if cata in model_cata_map:
                model_dir=model_cata_map[cata][0]
            else:
                model_dir=os.path.join(path_models_root, cata)
            full_path_file = os.path.abspath(os.path.join(model_dir, file_name))
            if os.path.exists(full_path_file):
                continue
            logger.info(f'The model file is not exists, ready to download: {file_name}')
            model_dir = os.path.dirname(full_path_file)
            file_name = os.path.basename(full_path_file)
            if url is None or url == '':
                url = f'{default_download_url_prefix}/{cata}/{path_file}'
            if path_file[:1]=='[' and path_file[-1:]==']' and url.endswith('.zip'):
                if not async_task:
                    download_diffusers_model_sync(cata, path_file[1:-1], size, url)
                else:
                    params=(dict(cata=cata, path_file=path_file[1:-1], num=size, url=url))
                    download_task = ('diffusers', params)
                    download_task_list.append(download_task)
            else:
                if not async_task:
                    load_file_from_url(
                        url=url,
                        model_dir=model_dir,
                        file_name=file_name
                    )
                else:
                    load_file_from_url(
                        url=url,
                        model_dir=model_dir,
                        file_name=file_name,
                        async_task=True
                    )
                    params=(dict(url=url, path_file=full_path_file))
                    download_task = ('file', params)
                    download_task_list.append(download_task)
        if async_task:
            pass #logger.info(f'download_task_list:{download_task_list}')
            #download_queue.put(lambda: download_multiple_files(download_task_list))
    return

async def download_diffusers_model(cata, model_name, num, url):
    download_diffusers_model_sync(cata, model_name, num, url)


def download_diffusers_model_sync(cata, model_name, num, url):
    import zipfile
    import shutil
    from modules.config import path_models_root, model_cata_map

    path_filter = [f'{model_name}/']
    result = shared.modelsinfo.get_model_names(cata, path_filter, casesensitive=True)
    if result is None or len(result)<num:
        path_temp = os.path.join(path_models_root, 'temp')
        if not os.path.exists(path_temp):
            os.makedirs(path_temp)
        file_name = os.path.basename(urlparse(url).path)
        load_file_from_url(
            url=url,
            model_dir=path_temp,
            file_name=file_name
        )
        downfile = os.path.join(path_temp, file_name)
        with zipfile.ZipFile(downfile, 'r') as zipf:
            logger.info(f'extractall: {downfile} to {path_temp}')
            zipf.extractall(path_temp)
        shutil.move(os.path.join(path_temp, f'SimpleModels/{cata}/{model_name}'), os.path.join(model_cata_map[cata][0], model_name))
        os.remove(downfile)
        shutil.rmtree(path_temp)
    shared.modelsinfo.refresh_from_path()
    return

def downloader(task_queue):
    while True:
        task = task_queue.get()
        logger.info('got download task')
        anyio.run(task)
        task_queue.task_done()

download_queue = queue.Queue()

#thread = threading.Thread(target=downloader, args=(download_queue,))
#thread.start()
