import ldm_patched.modules.args_parser as args_parser

args_parser.parser.add_argument("--share", action='store_true', help="Set whether to share on Gradio.")
args_parser.parser.add_argument("--preset", type=str, default='default', help="Apply specified UI preset.")
args_parser.parser.add_argument("--disable-preset-selection", action='store_true', default=True,
                                help="Disables preset selection in Gradio.")

args_parser.parser.add_argument("--language", type=str, default='cn',
                                help="Translate UI using json files in [language] folder. "
                                  "For example, [--language example] will use [language/example.json] for translation.")

args_parser.parser.add_argument("--webroot", type=str, default='', help="Set the webroot path.")
args_parser.parser.add_argument("--location", type=str, default='CN', help="Set the location access location.")

# For example, https://github.com/lllyasviel/Fooocus/issues/849
args_parser.parser.add_argument("--disable-offload-from-vram", action="store_true",
                                help="Force loading models to vram when the unload can be avoided. "
                                  "Some Mac users may need this.")

args_parser.parser.add_argument("--theme", type=str, help="Launches the UI with light or dark theme", default='dark')
args_parser.parser.add_argument("--disable-image-log", action='store_true',
                                help="Prevent writing images and logs to the outputs folder.")

args_parser.parser.add_argument("--disable-analytics", action='store_true',
                                help="Disables analytics for Gradio.")

args_parser.parser.add_argument("--disable-metadata", action='store_true',
                                help="Disables saving metadata to images.")

args_parser.parser.add_argument("--disable-preset-download", action='store_true',
                                help="Disables downloading models for presets", default=False)

args_parser.parser.add_argument("--disable-enhance-output-sorting", action='store_true',
                                help="Disables enhance output sorting for final image gallery.")

args_parser.parser.add_argument("--enable-auto-describe-image", action='store_true',
                                help="Enables automatic description of uov and enhance image when prompt is empty", default=True)

args_parser.parser.add_argument("--always-download-new-model", action='store_true',
                                help="Always download newer models", default=False)

args_parser.parser.add_argument("--rebuild-hash-cache", help="Generates missing model and LoRA hashes.",
                                type=int, nargs="?", metavar="CPU_NUM_THREADS", const=-1)

args_parser.parser.add_argument("--dev", action='store_true',
                                help="launch the dev branch", default=False)
args_parser.parser.add_argument("--models-root", type=str, help="Set the path root of models", default=None)
args_parser.parser.add_argument("--config", type=str, help="Set the path of config.txt", default=None)

args_parser.parser.add_argument("--disable-comfyd", action='store_true',
                                help="disable auto start comfyd server at launch", default=False)

args_parser.parser.add_argument("--reserve-vram", type=float, default=None, help="Set the amount of vram in GB you want to reserve for use by your OS/other software. By default some amount is reverved depending on your OS.")

args_parser.parser.set_defaults(
    disable_cuda_malloc=True,
    in_browser=True,
    port=None
)

args_parser.args = args_parser.parser.parse_args()

# (Disable by default because of issues like https://github.com/lllyasviel/Fooocus/issues/724)
args_parser.args.always_offload_from_vram = not args_parser.args.disable_offload_from_vram

if args_parser.args.disable_analytics:
    import os
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

if args_parser.args.disable_in_browser:
    args_parser.args.in_browser = False

args = args_parser.args
