import os
import json
import hashlib
import gradio as gr
import numbers
import copy
import re
import args_manager
import random
import cv2
import numpy as np
import base64
import shared
import modules.util as util
import modules.config as config
import modules.flags
import modules.sdxl_styles
import modules.constants as constants
import modules.meta_parser as meta_parser
import modules.sdxl_styles as sdxl_styles
import modules.style_sorter as style_sorter
import enhanced.all_parameters as ads
import enhanced.gallery as gallery_util
import enhanced.superprompter as superprompter
import enhanced.comfy_task as comfy_task
import ldm_patched.modules.model_management
import extras.preprocessors as preprocessors

from datetime import datetime
from modules.model_loader import load_file_from_url, presets_model_list, refresh_model_list, check_models_exists, download_model_files
from modules.private_logger import get_current_html_path
from modules.meta_parser import get_welcome_image
from enhanced.simpleai import comfyd, get_path_in_user_dir, toggle_identity_dialog
from enhanced.minicpm import minicpm, MiniCPM, is_chinese
from simpleai_base.simpleai_base import export_identity_qrcode_svg, import_identity_qrcode, gen_ua_session

# app context
nav_name_list = ''
system_message = ''
config_ext = {}
enhanced_config = os.path.abspath(f'./enhanced/config.json')
if os.path.exists(enhanced_config):
    with open(enhanced_config, "r", encoding="utf-8") as json_file:
        config_ext.update(json.load(json_file))
else:
    config_ext.update({'fooocus_line': '# 2.1.852', 'simplesdxl_line': '# 2023-12-20'})


def get_preset_name_list(user_did=None):
    if user_did and not shared.token.is_guest(user_did):
        user_preset_file = get_path_in_user_dir('presets.txt', user_did, 'presets')
        if not os.path.exists(user_preset_file):
            path_preset = os.path.abspath(f'./presets/')
            presets = [p for p in util.get_files_from_folder(path_preset, ['.json'], None) if not p.startswith('.')]
            file_times = [(f[:-5], os.path.getmtime(os.path.join(path_preset, f))) for f in presets]
            user_path_preset = get_path_in_user_dir('presets', user_did)
            if os.path.exists(user_path_preset):
                presets2 = [p for p in util.get_files_from_folder(user_path_preset, ['.json'], None) if not p.startswith('.')]
                file_times2 = [(f'{f[:-5]}.', os.path.getmtime(os.path.join(user_path_preset, f))) for f in presets2]
                file_times = file_times + file_times2
            presets = sorted(file_times, key=lambda x: x[1], reverse=True)
            presets = [f[0] for f in presets]
            if config.preset in presets:
                presets.remove(config.preset)
            presets.insert(0, config.preset)
            presets = presets[:shared.BUTTON_NUM]
            presets_list = ','.join(presets)
            with open(user_preset_file, 'w', encoding="utf-8") as nav_preset_file:
                nav_preset_file.write(presets_list)
        else:
            with open(user_preset_file, 'r', encoding="utf-8") as nav_preset_file:
                presets_list = nav_preset_file.read()
    else:
        guest_preset_file = get_path_in_user_dir('presets.txt', catalog='presets')
        if os.path.exists(guest_preset_file):
            with open(guest_preset_file, 'r', encoding="utf-8") as nav_preset_file:
                presets_list = nav_preset_file.read()
        else:
            path_preset = os.path.abspath(f'./presets/')
            presets = [p for p in util.get_files_from_folder(path_preset, ['.json'], None) if not p.startswith('.')]
            file_times = [(f[:-5], os.path.getmtime(os.path.join(path_preset, f))) for f in presets]
            presets = sorted(file_times, key=lambda x: x[1], reverse=True)
            presets = [f[0] for f in presets]
            if config.preset in presets:
                presets.remove(config.preset)
            presets.insert(0, config.preset)
            presets = presets[:shared.BUTTON_NUM]
            presets_list = ','.join(presets)
    return presets_list


preset_samples = {}
def get_preset_samples(user_did=None):
    global preset_samples
    path_preset = os.path.abspath(f'./presets/')
    presets = [p[:-5] for p in util.get_files_from_folder(path_preset, ['.json'], None) if not p.startswith('.')]
    if user_did and not shared.token.is_guest(user_did):
        user_path_preset = get_path_in_user_dir('presets', user_did)
        if os.path.exists(user_path_preset):
            presets2 = [p for p in util.get_files_from_folder(user_path_preset, ['.json'], None) if not p.startswith('.')]
            presets2 = [f'{p[:-5]}.' for p in presets2]
            presets = presets + presets2
    presets = sorted(presets)
    refresh_model_list(presets, user_did)
    presets.remove(config.preset)
    presets = [[p] for p in presets]
    if user_did:
        preset_samples[user_did] = presets
    else:
        preset_samples['guest'] = presets
    return presets

def is_models_file_absent(preset_name, user_did=None):
    if preset_name in presets_model_list:
        if check_models_exists(preset_name, user_did):
            return False
        else:
            return True
    preset_path = os.path.abspath(f'./presets/{preset_name}.json')
    if os.path.exists(preset_path):
        with open(preset_path, "r", encoding="utf-8") as json_file:
            config_preset = json.load(json_file)
        if config_preset["default_model"] and config_preset["default_model"] != 'None':
            if 'Flux' in preset_name and config_preset["default_model"]== 'auto':
                config_preset["default_model"] = comfy_task.get_default_base_Flux_name('+' in preset_name)
            model_key = f'checkpoints/{config_preset["default_model"]}'
            return not shared.modelsinfo.exists_model(catalog="checkpoints", model_path=config_preset["default_model"])
        if config_preset["default_refiner"] and config_preset["default_refiner"] != 'None':
           return not shared.modelsinfo.exists_model(catalog="checkpoints", model_path=config_preset["default_refiner"])
    return False


def get_system_message():
    global config_ext

    fooocus_log = os.path.abspath(f'./update_log.md')
    simplesdxl_log = os.path.abspath(f'./simplesdxl_log.md')
    update_msg_f = ''
    first_line_f = None
    if os.path.exists(fooocus_log):
        with open(fooocus_log, "r", encoding="utf-8") as log_file:
            line = log_file.readline()
            while line:
                if line == '\n':
                    line = log_file.readline()
                    continue
                if line.startswith("# ") and first_line_f is None:
                    first_line_f = line.strip()
                if line.strip() == config_ext['fooocus_line']:
                    break
                if first_line_f:
                    update_msg_f += line
                line = log_file.readline()
    update_msg_s = ''
    first_line_s = None
    if os.path.exists(simplesdxl_log):
        with open(simplesdxl_log, "r", encoding="utf-8") as log_file:
            line = log_file.readline()
            while line:
                if line == '\n':
                    line = log_file.readline()
                    continue
                if line.startswith("# ") and first_line_s is None:
                    first_line_s = line.strip()
                if line.strip() == config_ext['simplesdxl_line']:
                    break
                if first_line_s:
                    update_msg_s += line
                line = log_file.readline()
    update_msg_f = update_msg_f.replace("\n","  ")
    update_msg_s = update_msg_s.replace("\n","  ")
    
    f_log_path = os.path.abspath("./update_log.md")
    s_log_path = os.path.abspath("./simplesdxl_log.md")
    if len(update_msg_f)>0:
        body_f = f'<b id="update_f">[Fooocus更新信息]</b>: {update_msg_f}<a href="{args_manager.args.webroot}/file={f_log_path}">更多>></a>   '
    else:
        body_f = '<b id="update_f"> </b>'
    if len(update_msg_s)>0:
        body_s = f'<b id="update_s">[系统消息 - 已更新内容]</b>: {update_msg_s}<a href="{args_manager.args.webroot}/file={s_log_path}">更多>></a>'
    else:
         body_s = '<b id="update_s"> </b>'
    import mistune
    body = mistune.html(body_f+body_s)
    if first_line_f and first_line_s and (first_line_f != config_ext['fooocus_line'] or first_line_s != config_ext['simplesdxl_line']):
        config_ext['fooocus_line']=first_line_f
        config_ext['simplesdxl_line']=first_line_s
        with open(enhanced_config, "w", encoding="utf-8") as config_file:
            json.dump(config_ext, config_file)
    return body if body else ''



def preset_instruction():
    head = "<div style='max-width:100%; max-height:86px; overflow:hidden'>"
    foot = "</div>"
    body = '预置包简介:<span style="position: absolute;right: 0;"><a href="https://gitee.com/metercai/SimpleSDXL/blob/SimpleSDXL/presets/readme.md">\U0001F4DD 什么是预置包</a></span>'
    body += f'<iframe id="instruction" src="{get_preset_inc_url()}" frameborder="0" scrolling="no" width="100%"></iframe>'
    
    return head + body + foot

get_system_params_js = '''
function(system_params) {
    const params = new URLSearchParams(window.location.search);
    const sessionCookie = getCookie('aitoken');
    const url_params = Object.fromEntries(params);
    if (url_params["__lang"]) 
        system_params["__lang"]=url_params["__lang"];
    if (url_params["__theme"]) 
        system_params["__theme"]=url_params["__theme"];
    if (sessionCookie) 
        system_params["__session"]=sessionCookie;
    setObserver();
    return system_params;
}
'''

def init_nav_bars(state_params, comfyd_active_checkbox, fast_comfyd_checkbox, reserved_vram, minicpm_checkbox, advanced_logs, request: gr.Request):
    #print(f'request.headers:{request.headers}')
    #print(f'request.client:{request.client}')
    admin_currunt_value = [comfyd_active_checkbox, fast_comfyd_checkbox, reserved_vram, minicpm_checkbox, advanced_logs]

    if "__lang" not in state_params.keys():
        if 'accept-language' in request.headers and 'zh-CN' in request.headers['accept-language']:
            args_manager.args.language = 'cn'
        else:
            print(f'no accept-language in request.headers:{request.headers}')
        state_params.update({"__lang": args_manager.args.language}) 
    if "__theme" not in state_params.keys():
        state_params.update({"__theme": args_manager.args.theme})
    if "__preset" not in state_params.keys():
        state_params.update({"__preset": config.preset})
    user_agent = request.headers["user-agent"]
    ua_hash = hashlib.sha256(user_agent.encode('utf-8')).hexdigest()
    ua_session = gen_ua_session(request.client["host"], str(request.client["port"]), request.headers["user-agent"])
    state_params.update({"ua_hash": ua_hash})
    state_params.update({"ua_session": ua_session})
    if "__session" not in state_params.keys():
        sstoken = shared.token.get_guest_sstoken(ua_hash)
        state_params.update({"sstoken": sstoken})
        user_did = shared.token.get_guest_did()
        print(f'[UserBase] New request/新身份请求: {request.client.host}:{request.client.port} --> {request.headers.host} , ua_session={ua_session}')
    else:
        #print(f'aitoken: {state_params["__session"]}, guest={shared.token.get_guest_did()}')
        user_did = shared.token.check_sstoken_and_get_did(state_params["__session"], ua_hash)
        if user_did == "Unknown":
            sstoken = shared.token.get_guest_sstoken(ua_hash)
            state_params.update({"sstoken": sstoken})
            user_did = shared.token.get_guest_did()
            print(f'[UserBase] user-agent:{request.headers["user-agent"]}, cookie:{request.headers["cookie"]}')
            print(f'[UserBase] Reset request/重置的请求: {request.client.host}:{request.client.port} --> {request.headers.host} , ua_session={ua_session}')
        else:
            state_params.update({"sstoken": ''})
            print(f'[UserBase] Binded request/带身份请求: {request.client.host}:{request.client.port} --> {request.headers.host} , ua_session={ua_session}')
    state_params.update({"user": shared.token.get_user_context(user_did)})
    state_params.update({"user_did": user_did})
    state_params.update({"user_name":  shared.token.get_user_context(user_did).get_nickname()})
    state_params.update({"sys_did":  shared.token.get_sys_did()})

    if "__is_mobile" not in state_params.keys():
        state_params.update({"__is_mobile": True if user_agent.find("Mobile")>0 and user_agent.find("AppleWebKit")>0 else False})
    if "__webpath" not in state_params.keys():
        state_params.update({"__webpath": f'{args_manager.args.webroot}/file={os.getcwd()}'})
    if "__max_per_page" not in state_params.keys():
        if state_params["__is_mobile"]:
            state_params.update({"__max_per_page": 9})
        else:
            state_params.update({"__max_per_page": 18})
    if "__max_catalog" not in state_params.keys():
        state_params.update({"__max_catalog": config.default_image_catalog_max_number })
    state_params.update({"infobox_state": 0})
    state_params.update({"note_box_state": ['',0,0]})
    state_params.update({"array_wildcards_mode": '['})
    state_params.update({"wildcard_in_wildcards": 'root'})
    state_params.update({"bar_button": config.preset})
    state_params.update({"__nav_name_list": get_preset_name_list(user_did)})
    state_params.update({"preset_store": False})
    state_params.update({"engine": 'Fooocus'})
    results = [gr.update(value=f'{get_welcome_image(config.preset, state_params["__is_mobile"])}')]
    results += [gr.update(value=modules.flags.language_radio(state_params["__lang"])), gr.update(value=state_params["__theme"])]
    preset = 'default'
    preset_url = get_preset_inc_url(preset)
    state_params.update({"__preset_url":preset_url})
    results += [gr.update(visible=True if 'blank.inc.html' not in preset_url else False)]
    results += get_all_admin_default(admin_currunt_value)

    return results

def get_preset_inc_url(preset_name='blank'):
    preset_name = f'{preset_name}.inc'
    preset_inc_path = os.path.abspath(f'./presets/html/{preset_name}.html')
    blank_inc_path = os.path.abspath(f'./presets/html/blank.inc.html')
    if os.path.exists(preset_inc_path):
        return f'{args_manager.args.webroot}/file={preset_inc_path}'
    else:
        return f'{args_manager.args.webroot}/file={blank_inc_path}'

def refresh_nav_bars(state_params):
    preset_name_list = state_params["__nav_name_list"].split(',')
    for i in range(shared.BUTTON_NUM-len(preset_name_list)):
        preset_name_list.append('')
    results = []
    if state_params["__is_mobile"]:
        results += [gr.update(visible=False)]
    else:
        results += [gr.update(visible=True)]
    for i in range(len(preset_name_list)):
        name = preset_name_list[i]
        name += '\u2B07' if is_models_file_absent(name, state_params["user_did"]) else ''
        visible_flag = i<(7 if state_params["__is_mobile"] else shared.BUTTON_NUM)
        if name:
            results += [gr.update(value=name, interactive=True, visible=visible_flag)]
        else: 
            results += [gr.update(value='', interactive=False, visible=visible_flag)]
    return results


def avoid_empty_prompt_for_scene(prompt, state, img, scene_theme, additional_prompt):
    describe_prompt, img_is_ok = describe_prompt_for_scene(state, img, scene_theme, additional_prompt) if not prompt and 'scene_frontend' in state else None
    return gr.update() if describe_prompt is None else describe_prompt

def describe_prompt_for_scene(state, img, scene_theme, additional_prompt):
    img = img if img is None else util.resize_image(img, max_side=1280, resize_mode=4)
    image_preprocessor_method = state['scene_frontend'].get('image_preprocessor_method', [])
    img_is_ok = preprocessors.openpose_have(img, image_preprocessor_method[0]) if len(image_preprocessor_method)>0 and img is not None else True
    s_prompts = state['scene_frontend'].get('prompt', {})
    describe_prompt = s_prompts.get(scene_theme, '')
    if not describe_prompt:
        return '', img_is_ok
    describe_prompt = describe_prompt.format(additional_prompt=additional_prompt)
    m_prompts = state['scene_frontend'].get('multimodal_prompt', {})
    prompt_prompt = m_prompts.get(scene_theme, '')
    if prompt_prompt and img is not None:
        prompt_prompt = prompt_prompt.format(additional_prompt=additional_prompt)
        if MiniCPM.get_enable():
            describe_prompt += minicpm.interrogate(img, prompt=prompt_prompt)
        else:
            from extras.interrogate import default_interrogator as default_interrogator_photo
            describe_prompt += default_interrogator_photo(img)
            from extras.wd14tagger import default_interrogator as default_interrogator_anime
            describe_prompt += default_interrogator_anime(img)
    return describe_prompt, img_is_ok


def process_before_generation(state_params, backend_params, backfill_prompt, translation_methods, comfyd_active_checkbox, hires_fix_stop, hires_fix_weight, hires_fix_blurred, reserved_vram, scene_input_image1, scene_theme, scene_additional_prompt, scene_aspect_ratio, scene_image_number):
    superprompter.remove_superprompt()
    remove_tokenizer()
    minicpm.free_model()
    backend_params.update(dict(
        nickname=state_params["user_name"],
        user_did=state_params["user_did"],
        translation_methods=translation_methods,
        backfill_prompt=backfill_prompt,
        comfyd_active_checkbox=comfyd_active_checkbox,
        preset=state_params["__preset"],
        hires_fix_stop=hires_fix_stop,
        hires_fix_weight=hires_fix_weight,
        hires_fix_blurred=hires_fix_blurred,
        reserved_vram=reserved_vram
        ))
    
    if 'scene_frontend' in state_params:
        backend_params.update(dict(
            task_method=state_params['scene_frontend']['task_method'][scene_theme],
            scene_frontend=state_params['scene_frontend']['version'],
            scene_input_image1=scene_input_image1,
            scene_theme=scene_theme,
            scene_additional_prompt=scene_additional_prompt,
            scene_aspect_ratio=scene_aspect_ratio.split('|')[0] if '×' in scene_aspect_ratio else modules.flags.scene_aspect_ratios_size[scene_aspect_ratio],
            scene_image_number=scene_image_number
            ))

    if is_models_file_absent(state_params["__preset"], state_params["user_did"]):
        gr.Info(preset_downing_note_info)
        download_model_files(state_params["__preset"], state_params["user_did"])

    # stop_button, skip_button, generate_button, gallery, state_is_generating, index_radio, image_toolbox, prompt_info_box
    results = [gr.update(visible=True, interactive=True), gr.update(visible=True, interactive=True), gr.update(visible=False, interactive=False), [], True, gr.update(visible=False, open=False), gr.update(visible=False), gr.update(visible=False)]
    # random_button, translator_button, super_prompter, background_theme, image_tools_checkbox, bar_store_button, bar0_button, bar1_button, bar2_button, bar3_button, bar4_button, bar5_button, bar6_button, bar7_button, bar8_button
    preset_nums = len(state_params["__nav_name_list"].split(','))
    results += [gr.update(interactive=False)] * (preset_nums + 6)
    results += [gr.update()] * (shared.BUTTON_NUM-preset_nums)
    # preset_store, identity_dialog
    results += [gr.update(visible=False)]*2

    state_params["gallery_state"]='preview'
    state_params["preset_store"]=False
    state_params["identity_dialog"]=False
    return results


def process_after_generation(state_params):
    if "__max_per_page" not in state_params.keys():
        state_params.update({"__max_per_page": 18})
    if "__max_catalog" not in state_params.keys():
        state_params.update({"__max_catalog": config.default_image_catalog_max_number })
    
    max_per_page = state_params["__max_per_page"]
    max_catalog = state_params["__max_catalog"]
    user_did = state_params["user_did"]
    output_list, finished_nums, finished_pages = gallery_util.refresh_output_list(max_per_page, max_catalog, user_did)
    state_params.update({"__output_list": output_list})
    state_params.update({"__finished_nums_pages": f'{finished_nums},{finished_pages}'})
    # generate_button, stop_button, skip_button, state_is_generating
    results = [gr.update(visible=True, interactive=True)] + [gr.update(visible=False, interactive=False), gr.update(visible=False, interactive=False), False]
    # gallery_index, index_radio
    results += [gr.update(choices=state_params["__output_list"], value=None), gr.update(visible=len(state_params["__output_list"])>0, open=False)]
    # random_button, translator_button, super_prompter, background_theme, image_tools_checkbox, bar_store_button, bar0_button, bar1_button, bar2_button, bar3_button, bar4_button, bar5_button, bar6_button, bar7_button, bar8_button
    preset_nums = len(state_params["__nav_name_list"].split(','))
    results += [gr.update(interactive=True)] * (preset_nums + 6)
    results += [gr.update()] * (shared.BUTTON_NUM-preset_nums)
    # [history_link, gallery_index_stat]
    results += [state_params['__finished_nums_pages']]
    results += [update_history_link(user_did)]
    

    if len(state_params["__output_list"]) > 0:
        output_index = state_params["__output_list"][0].split('/')[0]
        gallery_util.refresh_images_catalog(output_index, True, user_did)
        gallery_util.parse_html_log(output_index, True, user_did)
   

    return results


def sync_message(state_params):
    state_params.update({"__message":system_message})
    return

preset_down_note_info = 'The preset package being loaded has model files that need to be downloaded.'
preset_downing_note_info = 'Downloading the model file required for image generation, please wait for a moment...'

def check_absent_model(bar_button, state_params):
    #print(f'check_absent_model,state_params:{state_params}')
    state_params.update({'bar_button': bar_button})
    return 

def down_absent_model(state_params):
    state_params.update({'bar_button': state_params["bar_button"].replace('\u2B07', '')})
    return gr.update(visible=False), state_params

reset_layout_num = 0

def reset_layout_params(prompt, negative_prompt, state_params, is_generating, inpaint_mode, comfyd_active_checkbox):
    global system_message, preset_down_note_info, reset_layout_num1, reset_layout_num2

    state_params.update({"__message": system_message})
    system_message = 'system message was displayed!'
    if '__preset' not in state_params.keys() or 'bar_button' not in state_params.keys() or state_params["__preset"]==state_params['bar_button']:
        return refresh_nav_bars(state_params) + [gr.update()] * reset_layout_num + update_after_identity_sub(state_params)
    preset = state_params["bar_button"] if '\u2B07' not in state_params["bar_button"] else state_params["bar_button"].replace('\u2B07', '')
    print(f'[Topbar] Reset_context: preset={state_params["__preset"]}-->{preset}, theme={state_params["__theme"]}, lang={state_params["__lang"]}')
    if '\u2B07' in state_params["bar_button"]:
        gr.Info(preset_down_note_info)
        #download_model_files(preset, state_params["user_did"])

    state_params.update({"__preset": preset})

    config_preset = config.try_get_preset_content(preset, state_params["user_did"])
    preset_prepared = meta_parser.parse_meta_from_preset(config_preset)
    preset_prepared.update({
        'preset': preset,
        'is_mobile': state_params["__is_mobile"] })
    
    #print(f'preset_prepared:{preset_prepared}')
    
    engine = preset_prepared.get('engine', {}).get('backend_engine', 'Fooocus')
    state_params.update({"engine": engine})
    scene_frontend = preset_prepared.get('engine', {}).get('scene_frontend', None)
    if scene_frontend:
        state_params.update({"scene_frontend": scene_frontend})
        task_method = scene_frontend['task_method']
        if isinstance(task_method, list):
            task_method = task_method[0]
        elif isinstance(task_method, dict):
            if task_method:
                task_method = task_method[next(iter(task_method))]
    else:
        if 'scene_frontend' in state_params:
            del state_params["scene_frontend"]

        task_method = preset_prepared.get('engine', {}).get('backend_params', modules.flags.get_engine_default_backend_params(engine))
    state_params.update({"task_method": task_method})
    
    if comfyd_active_checkbox:
        comfyd.stop()
   
    default_model = preset_prepared.get('base_model')
    previous_default_models = preset_prepared.get('previous_default_models', [])
    checkpoint_downloads = preset_prepared.get('checkpoint_downloads', {})
    embeddings_downloads = preset_prepared.get('embeddings_downloads', {})
    lora_downloads = preset_prepared.get('lora_downloads', {})
    vae_downloads = preset_prepared.get('vae_downloads', {})

    model_dtype = preset_prepared.get('engine', {}).get('backend_params', {}).get('base_model_dtype', '')
    if engine == 'SD3x' and  model_dtype == 'auto':
        base_model = comfy_task.get_default_base_SD3m_name()
        if shared.modelsinfo.exists_model(catalog="checkpoints", model_path=base_model):
            default_model = base_model
            preset_prepared['base_model'] = base_model
            checkpoint_downloads = {}
    if engine == 'Flux' and default_model=='auto':
        default_model = comfy_task.get_default_base_Flux_name('FluxS' in preset)
        preset_prepared['base_model'] = default_model
        if shared.modelsinfo.exists_model(catalog="checkpoints", model_path=default_model):
            checkpoint_downloads = {}
        else:
            checkpoint_downloads = {default_model: comfy_task.flux_model_urls[default_model]}
            if 'merged' in default_model:
                preset_prepared.update({'default_overwrite_step': 6})

    download_models(default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads, vae_downloads)

    preset_url = preset_prepared.get('reference', get_preset_inc_url(preset))
    state_params.update({"__preset_url":preset_url})

    results = refresh_nav_bars(state_params)
    results += meta_parser.switch_layout_template(preset_prepared, state_params, preset_url)
    results += meta_parser.load_parameter_button_click(preset_prepared, is_generating, inpaint_mode)
    results += update_after_identity_sub(state_params)

    ldm_patched.modules.model_management.print_memory_info("after switched preset")
    return results


def download_models(default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads, vae_downloads):

    if shared.args.disable_preset_download:
        print('Skipped model download.')
        return default_model, checkpoint_downloads

    if not shared.args.always_download_new_model:
        if not os.path.isfile(shared.modelsinfo.get_file_path_by_name('checkpoints', default_model)):
            for alternative_model_name in previous_default_models:
                if os.path.isfile(shared.modelsinfo.get_file_path_by_name('checkpoints', alternative_model_name)):
                    print(f'You do not have [{default_model}] but you have [{alternative_model_name}].')
                    print(f'Fooocus will use [{alternative_model_name}] to avoid downloading new models, '
                          f'but you are not using the latest models.')
                    print('Use --always-download-new-model to avoid fallback and always get new models.')
                    checkpoint_downloads = {}
                    default_model = alternative_model_name
                    break

    for file_name, url in checkpoint_downloads.items():
        model_dir = os.path.dirname(shared.modelsinfo.get_file_path_by_name('checkpoints', file_name))
        load_file_from_url(url=url, model_dir=model_dir, file_name=os.path.basename(file_name))
    for file_name, url in embeddings_downloads.items():
        load_file_from_url(url=url, model_dir=config.path_embeddings, file_name=file_name)
    for file_name, url in lora_downloads.items():
        model_dir = os.path.dirname(shared.modelsinfo.get_file_path_by_name('loras', file_name))
        load_file_from_url(url=url, model_dir=model_dir, file_name=os.path.basename(file_name))
    for file_name, url in vae_downloads.items():
        load_file_from_url(url=url, model_dir=config.path_vae, file_name=file_name)

    return default_model, checkpoint_downloads

def toggle_preset_store(state):
    if 'user_did' in state and not shared.token.is_guest(state["user_did"]):
        if 'preset_store' in state:
            flag = state['preset_store']
        else:
            state['preset_store'] = False
            flag = False
        state['preset_store'] = not flag
        state['identity_dialog'] = False
        return [gr.update(visible=not flag)] + update_topbar_js_params(state) + [gr.update(visible=False)] + [gr.update()]*16
    else:
        #state['identity_dialog'] = False
        return [gr.update()] + update_topbar_js_params(state) + toggle_identity_dialog(state)

def update_navbar_from_mystore(selected_preset, state):
    global preset_samples
    selected_preset = preset_samples[state['user_did'] if not shared.token.is_guest(state["user_did"]) else 'guest'][selected_preset][0]
    results = refresh_nav_bars(state)
    results2 = update_topbar_js_params(state)
    nav_list_str = state["__nav_name_list"]
    nav_array = nav_list_str.split(',')
    if selected_preset in ["default", state["__preset"]]:
        return results + results2
    if selected_preset in nav_array:
        nav_array.remove(selected_preset)
        print(f'[PresetStore] Withdraw the preset/回撤预置包: {selected_preset}.')
    else:
        if len(nav_array) >= shared.BUTTON_NUM:
            return results + results2
        nav_array.append(selected_preset)
        print(f'[PresetStore] Launch the preset/启用预置包: {selected_preset}.')
    state["__nav_name_list"] = ','.join(nav_array)
    if 'user_did' in state and not shared.token.is_guest(state["user_did"]):
        user_preset_file = get_path_in_user_dir('presets.txt', state["user_did"], 'presets')
        with open(user_preset_file, 'w', encoding="utf-8") as nav_preset_file:
            nav_preset_file.write(state["__nav_name_list"])
    #print(f'__nav_name_list:{state["__nav_name_list"]}')
    return refresh_nav_bars(state) + update_topbar_js_params(state)

def admin_sync_to_guest(state, catalog='presets'):
    user_did = state["user"].get_did()
    if shared.token.is_admin(user_did):
        if catalog == 'presets':
            user_preset_file = get_path_in_user_dir('presets.txt', user_did, 'presets')
            guest_preset_file = get_path_in_user_dir('presets.txt', shared.token.get_guest_did(), 'presets')
            shutil.copy(user_preset_file, guest_preset_file)
    current_time = datetime.now().strftime("%H:%M:%S")
    admin_sync_title = 'Sync presets nav to guest' if state["__lang"]!='cn' else '同步预置导航给游客'
    print(f'[Topbar] Sync presets nav to guest: {current_time}')
    return f'{admin_sync_title}({current_time})'



def update_topbar_js_params(state):
    system_params= dict(
        __preset=state["__preset"],
        __theme=state["__theme"],
        __nav_name_list=state["__nav_name_list"],
        sstoken=state["sstoken"],
        user_name=state["user_name"],
        user_did=state["user_did"],
        user_role='guest' if shared.token.is_guest(state["user_did"]) else 'admin' if shared.token.is_admin(state["user_did"]) else 'member',
        task_class_name=state["engine"],
        preset_store=state["preset_store"],
        __message=state["__message"],
        __webpath=state["__webpath"],
        __lang=state["__lang"],
        __preset_url=state["__preset_url"],
        __finished_nums_pages=state["__finished_nums_pages"],
        user_qr="" if 'user_qr' not in state else state["user_qr"]
        )
    return [system_params]


def export_identity(state):
    if not shared.token.is_guest(state["user_did"]):
        state["user_qr"] = export_identity_qrcode_svg(state["user_did"])
        #print(f'user_qrcode_svg: {state["user_qr"]}')
    return update_topbar_js_params(state)[0]

def trigger_input_identity(img):
    image = util.HWC3(img)
    qr_code_detector = cv2.QRCodeDetector()
    data, bbox, _ = qr_code_detector.detectAndDecode(image)
    if bbox is not None:
        try:
            user_did, nickname, telephone = import_identity_qrcode(data)
        except Exception as e:
            print("qrcode parse error")
            user_did, nickname, telephone = '', '', ''
    else:
        user_did, nickname, telephone = '', '', ''
    return [gr.update(visible=False), gr.update(visible=True), f'{nickname}, {telephone}']

def update_history_link(user_did):
    return gr.update(value='' if args_manager.args.disable_image_log else f'<a href="file={get_current_html_path(None, user_did)}" target="_blank">\U0001F4DA History Log</a>')

def update_comfyd_url(user_did):
    entry_point = '' if comfyd.get_entry_point_id() is None else shared.token.get_entry_point(user_did, comfyd.get_entry_point_id())
    entry_url = None if entry_point == '' else f'http://{args_manager.args.listen}:{shared.sysinfo["loopback_port"]}{args_manager.args.webroot}/'
    entry_point_url = '' if entry_url is None else f'<a href="{entry_url}?p={entry_point}" target="_blank">{entry_url}</a><div>Click and Entry embedded ComfyUI from here.</div>'
    return entry_point_url
   
identity_introduce = '''
当前为游客，点击"身份管理"绑定身份，解锁更多功能：<br>
1，解锁“我的预置”功能，支持个性化的预置导航。<br>
2，独立的出图存储空间和日志历史页，保障隐私安全。<br>
3，可将当前环境参数保存为个人定制的预置包。<br>
4，解锁更多功能的配置管理，包括翻译器/定制OBP等。<br>
5，其他计划中的个性化服务、增强功能及互助服务。<br>
如：提示词库、创意分享、共享预置包、共享算力等<br>
<br>
系统将指定首个绑定身份为管理员，赋予超级管理权限: <br>
1，可一键进入内嵌的Comfyd引擎工作流操作界面。<br>
2，可管理内嵌Comfyd引擎的更多性能参数和配置。<br>
3，解锁MiniCPM，升级更高级的反推/翻译/扩写服务。<br>
4，对本节点的其他身份进行管理(计划)。<br>
5，申请预置包发布和二次打包的授权标识(计划)。<br>
关注计划中需求的可以入QQ群:938075852 进行交流。<br>
<br>
系统遵循分布式身份管理机制，即用户自主掌控身份私钥，授权AI节点使用身份；AI节点私有部署，管理多用户相互隔离的数字空间；社区节点保存加密副本用于追溯和自证。在多方协作下共同保障隐私安全、身份可信及跨节点互认。以此构建"和而不同"的开源社区生态。详细说明>> <br>
'''

def update_after_identity(state):
    state.update({"__nav_name_list": get_preset_name_list(state["user_did"])})
    results = refresh_nav_bars(state)
    results += update_after_identity_sub(state)
    return results

def update_after_identity_sub(state):
    #[gallery_index, index_radio, gallery_index_stat, layer_method, layer_input_image, preset_store, preset_store_list, history_link, identity_introduce, admin_panel, admin_link, user_panel, system_params]
    max_per_page = state["__max_per_page"]
    max_catalog = state["__max_catalog"]
    nickname = state["user_name"]
    user_did = state["user_did"]
    print(f'[UserBase] Session identity/当前身份: {nickname}({user_did}{", admin" if shared.token.is_admin(user_did) else ""}), ua_session({state["ua_session"]})')
    output_list, finished_nums, finished_pages = gallery_util.refresh_output_list(max_per_page, max_catalog, user_did)
    state.update({"__output_list": output_list})
    state.update({"__finished_nums_pages": f'{finished_nums},{finished_pages}'})

    if shared.token.is_admin(user_did):
        admin_outputs = os.path.join(shared.token.get_path_in_user_dir(user_did, "outputs"), 'ComfyUI')
        if not os.path.exists(admin_outputs):
            os.makedirs(admin_outputs)
        comfyd.modify_variable({"outputs": admin_outputs})

    results = [gr.update(choices=output_list, value=None), gr.update(visible=len(output_list)>0, open=False)]
    results += [state['__finished_nums_pages']]
    results += [gr.update(interactive=True if state["engine"]=='Fooocus' and not shared.token.is_guest(user_did) else False)] *2
    results += [gr.update(visible=False if 'preset_store' not in state else state['preset_store'])]
    results += [gr.Dataset.update(samples=get_preset_samples(user_did))]
    results += [update_history_link(user_did)]
    results += [gr.update(visible=shared.token.is_guest(user_did))]
    results += [gr.update(visible=shared.token.is_admin(user_did))]
    results += [gr.update(value=update_comfyd_url(user_did))]
    results += [gr.update(visible=not shared.token.is_guest(user_did))]
    results += update_topbar_js_params(state)
    ip_list = modules.flags.ip_list if state["engine"] in ['Fooocus', 'Flux', 'Kolors', 'Comfy']  else modules.flags.ip_list[:-1]
    default_controlnet_image_count = config.default_controlnet_image_count if state["engine"]=='Fooocus' else 4
    for image_count in range(default_controlnet_image_count):
        image_count += 1
        results.append(gr.update(choices=ip_list, value=config.default_ip_types[image_count]))

    return results

def update_size_and_hires_fix(image, uov_method, params_backend, hires_fix_stop, hires_fix_weight, hires_fix_blurred):
    size_image = update_upscale_size_of_image(image, uov_method)
    params_backend.update({'i2i_uov_hires_fix_s': hires_fix_stop})
    params_backend.update({'i2i_uov_hires_fix_w': hires_fix_weight})
    params_backend.update({'i2i_uov_hires_fix_blurred': hires_fix_blurred})
    vary_strength = -1
    upscale_strength = -1
    #if 'Upscale' in uov_method:
    #    upscale_strength = 0.2
    #if 'Subtle' in uov_method:
    #    vary_strength = 0.5
    #if 'Strong' in uov_method:
    #    vary_strength = 0.85
    if 'Hires.fix' in uov_method:
        vary_strength = 0.85
        return gr.update(visible=False, value=size_image), gr.update(visible=True), vary_strength, upscale_strength
    else:
        return gr.update(visible=True, value=size_image), gr.update(visible=False), vary_strength, upscale_strength

def update_upscale_size_of_image(image, uov_method):
    if image is not None:
        H, W, C = util.HWC3(image).shape
    else:
        return ''
    match = re.search(r'\((?:Fast )?([\d.]+)x\)', uov_method)
    match_multiple = 1.0 if not match else float(match.group(1))
    match_multiple = match_multiple if match_multiple<4.0 else 4.0 
    width = int(W * match_multiple)
    height = int(H * match_multiple)

    return f'{W} x {H} | {width} x {height}'


def admin_save_to_default(state, comfyd_active_checkbox, fast_comfyd_checkbox, reserved_vram, minicpm_checkbox, advanced_logs):
    set_admin_default_value = lambda x,y,s: shared.token.set_local_vars(f'admin_{x}', str(y), s["__session"], s["ua_hash"])
    
    set_admin_default_value('comfyd_active_checkbox', comfyd_active_checkbox, state)
    set_admin_default_value('fast_comfyd_checkbox', fast_comfyd_checkbox, state)
    set_admin_default_value('reserved_vram', reserved_vram, state)
    set_admin_default_value('minicpm_checkbox', minicpm_checkbox, state)
    set_admin_default_value('advanced_logs', advanced_logs, state)

    current_time = datetime.now().strftime("%H:%M:%S")
    admin_save_title = 'Save default of system' if state["__lang"]!='cn' else '保存系统默认值'
    print(f'[Topbar] Save admin config to default: {current_time}')
    return f'{admin_save_title}({current_time})'


def get_all_admin_default(currunt_value):
    admin_keys = ['comfyd_active_checkbox', 'fast_comfyd_checkbox', 'reserved_vram', 'minicpm_checkbox', 'advanced_logs']
    result = []
    for i, admin_key in enumerate(admin_keys): 
        admin_value = ads.get_admin_default(admin_key)
        if admin_value == 'None':
            result.append(gr.update(interactive=False))
            continue
        if admin_value == currunt_value[i]:
            result.append(gr.update())
        else:
            result.append(gr.update(interactive=True, value=admin_value))

    return result


def get_auto_candidate(img, selections, mode):
    H, W, C = img.shape
    selections2 = [ x.split('|')[1] if '|' in x else x for x in selections ]
    selections2 = [ float(x.split(':')[0])/float(x.split(':')[1]) for x in selections2 ]
    selection = float(W)/float(H)
    selections2 = np.array(selections2)
    index = np.argmin(np.abs(selections2 - selection))
    index_value = selections[index]
    if '_candidate' in mode:
        start_index = max(index - 1, 0)
        end_index = min(index + 2, len(selections2))
        selections = selections[start_index:end_index]
    return selections, index_value


from transformers import CLIPTokenizer
import shutil

cur_clip_path = os.path.join(config.path_clip_vision, "clip-vit-large-patch14")
if not os.path.exists(cur_clip_path):
    org_clip_path = os.path.join(shared.root, 'models/clip_vision/clip-vit-large-patch14')
    shutil.copytree(org_clip_path, cur_clip_path)
tokenizer = CLIPTokenizer.from_pretrained(cur_clip_path)
 
def remove_tokenizer():
    global tokenizer

    if 'tokenizer' in globals():
        del tokenizer
    return

def prompt_token_prediction(text, style_selections):
    global tokenizer, cur_clip_path
    if 'tokenizer' not in globals():
        globals()['tokenizer'] = None
    if tokenizer is None:
        tokenizer = CLIPTokenizer.from_pretrained(cur_clip_path)
    return len(tokenizer.tokenize(text))

nav_name_list = get_preset_name_list()
system_message = get_system_message()
