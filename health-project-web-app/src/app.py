from h2o_wave import main, app, Q, ui, on, run_on, copy_expando
from loguru import logger
from .layout import add_card, clear_cards, init
from .constants import PREDICTION_ANIMATION
from .model_utils import predict_models, download_data_from_s3


@app('/')
async def serve(q: Q):
    if not q.app.initialized:
        await initialize_app(q)
    if not q.client.initialized:
        q.client.cards = set()
        await init(q)
        if q.args['#'] is None:
            await page1(q)
        q.client.initialized = True
        q.client.stepper = 0
        q.client.file_name = None
        q.client.model_path = "models/l1_ssim baseline_no_roi.ckpt"

    logger.warning(f'============\n  q.args: {q.args}')
    copy_expando(q.args, q.client)
    await run_on(q)
    await q.page.save()


async def initialize_app(q):
    q.app.loader, = await q.site.upload([PREDICTION_ANIMATION])
    q.app.initialized = True


@on('#page1')
async def page1(q: Q):
    q.page['sidebar'].value = '#page1'
    image, = await q.site.upload(['static/home_image.jpg'])
    image_markup = ui.markup(f'<center><img style="width:100%;" src="{image}"/></center>')
    clear_cards(q)
    add_card(q, 'article', ui.form_card(
        box=ui.box('vertical', height='100%', width='90%'),
        items=[
            image_markup,
            ui.text_l('**A Novel Approach for MRI Reconstruction Acceleration Using Deep Learning**'),
            ui.text('''This project introduces an innovative approach to medical image reconstruction by integrating deep learning techniques with region-aware supervision to enhance MRI image quality.

Our methodology includes:
- **Deep Learning & ROI-Aware Architectures:** Leveraging convolutional neural networks like U-Net, augmented with attention mechanisms (e.g., CBAM), to reconstruct high-fidelity MRI images while prioritizing clinically relevant regions.
- **Annotation-Guided Learning:** Using enriched datasets with expert-provided annotations to guide the model’s focus during training, improving reconstruction accuracy in diagnostically critical areas.
- **Evaluation & Deployment:** A robust evaluation pipeline for region-specific metrics (SSIM, PSNR, NMSE), and a web-based application enabling real-time visualization and model comparison.

By combining medical imaging expertise with advanced machine learning and interactive tools, this project aims to support more accurate and efficient diagnostics in radiology workflows.
'''),
        ]
    ))


@on('#page2')
@on('stepper_prediction')
async def page2(q: Q):
    q.page['sidebar'].value = '#page2'
    clear_cards(q)
    q.client.stepper = 0
    items = await stepper_items(q)
    add_card(q, 'stepper_infos',
            ui.form_card(
                box=ui.box('horizontal', height='900px', width='100%'),
                title='',
                items=items
                ))


@on('file_upload')
async def file_upload(q: Q):
    q.client.file_path = await q.site.download(url=q.args.file_upload[0], path='./data')
    q.client.file_name = q.args.file_upload[0].split('/')[-1]
    q.page['meta'].dialog = None
    clear_cards(q)
    q.client.stepper = 1
    items = await stepper_items(q)
    add_card(q, 'stepper_infos',
            ui.form_card(
                box=ui.box('horizontal', height='900px', width='100%'),
                title='',
                items=items
                ))


async def stepper_items(q):
    items = [
            ui.stepper(name='stepper_prediction', items=[
                        ui.step(label='Upload MRI Input', done=True if q.client.stepper > 0 else False, icon='BulkUpload'),
                        ui.step(label='Choose Model', icon='IOT', done=True if q.client.stepper > 1 else False),
                        ui.step(label='Perform Prediction', icon='MachineLearning', done=True if q.client.stepper >= 2 else False),
                        ui.step(label='Analyze & Download Images', icon='ImageSearch'),
                    ])
            ]
    if q.client.stepper == 0:
        # demo_files = ['data/file1002549.h5']
        files_upload = items + [
            ui.inline(
                [
                    # sep,
                    ui.text_xl('''Upload MRI Input'''),
                    ui.text('''Please upload an .h5 file containing an undersampled MRI image in the frequency domain. The file should include a multi-channel array of shape [C, H, W]'''),
                    ui.file_upload(name='file_upload', label='Input File Upload', multiple=False, file_extensions=['h5'], width='min(95%, 450px)',),
                    ui.separator(width='50%', name='separator_info_tabs', label='or upload demo file'),
                    ui.dropdown(name='demo_file', label='Demo File', choices=[
                        ui.choice(name=demo_file, label=demo_file) for demo_file in h5_files
                    ], placeholder='Select a demo file', width='min(95%, 450px)', trigger=True),
                    ui.button(name='demo_upload', label='Upload File', primary=True, disabled=True),
                ], justify='center', align='center', direction='column'),
            ]
        return files_upload
    elif q.client.stepper == 1:
        model_choices = [ui.choice(name=k, label=v) for k, v in models_names.items()]
        items += [
            ui.inline(
                items=[
                    ui.text_xl('''Model Selection'''),
                    ui.picker(
                        name='model_picker',
                        label='Select MRI Reconstruction Models to use',
                        choices=model_choices,
                        values=["unet_baseline_l1_ssim_l1_ssim", "unet_full_attention_l1_ssim_roi_l1_ssim_roi_attn_agate"],
                        width='min(95%, 450px)',
                        ),
                    ui.text("""
Our project implements several deep learning models for MRI reconstruction that you can choose, each with unique characteristics:

## Baseline U-Net
The standard convolutional network architecture widely used for image segmentation and reconstruction tasks. This model serves as our performance benchmark, featuring an encoder-decoder structure with skip connections that help preserve spatial information.

## U-Net with ROI-weighted Loss
This model extends the baseline U-Net by incorporating our region-specific loss function that prioritizes clinically significant areas. The specialized loss function weights reconstruction errors in annotated regions more heavily than errors in less diagnostic regions.

## CBAM-enhanced U-Net with and without ROI-weighted Loss
Integrates the Convolutional Block Attention Module into the U-Net architecture. CBAM combines channel attention (learning "what" features are important) and spatial attention (learning "where" features are important) to improve focus on critical areas without explicit annotation guidance.

## Attention Gates U-Net with and without ROI-weighted Loss
Modifies the skip connections in U-Net to selectively focus on relevant features from the encoder path before concatenation in the decoder path. This reduces irrelevant feature propagation and improves reconstruction quality in complex regions.

## Full Attention U-Net with and without ROI-weighted Loss
Our most advanced model, combining both CBAM within convolutional blocks and Attention Gates on skip connections. This comprehensive approach to attention mechanisms provides maximum flexibility in focusing computational resources on diagnostically relevant features.
                    """),
                    ui.button(name='predict', label='Perform Prediction', primary=True),
                ],
                justify='center', align='center', direction='column'
            ),
        ]
        return items
    elif q.client.stepper == 2:
        if q.client.predictions_files:
            imgs_markups = []
            input_image, = await q.site.upload([f'predictions/input_{q.client.file_name}.png'])
            imgs_markups.append(ui.inline(
                items=[
                    ui.markup('''
<div style="text-align: center; margin-right: 100px; border: 2px solid red; border-radius: 8px; padding: 10px;">
  <h2 style="color: #2c3e50; font-size: 18px; margin-bottom: 10px;">Input Image</h2>
  <img style="width:250px; border: 1px solid #e0e0e0; border-radius: 4px;" src="{}"/>
</div>
'''.format(input_image))
                ],
                justify='center', align='center', direction='column'
            ))
            for i, prediction_file in enumerate(q.client.predictions_files):
                prediction_image, = await q.site.upload([prediction_file])
                model_name = prediction_file.split('/')[-1].split('.')[0]
                model_name = '_'.join(model_name.split('_')[:-1])
                model_name = models_names.get(model_name, f'Model {i+1}')
                imgs_markups.append(
                    ui.inline(
                items=[
                    ui.markup('''
<div style="text-align: center; margin-right: 20px;">
  <h2 style="color: #2c3e50; font-size: 15px; margin-bottom: 10px;">{}</h2>
  <p style="color: #7f8c8d; font-style: italic; margin-bottom: 12px;">MRI Reconstruction</p>
  <img style="width:250px; border: 1px solid #e0e0e0; border-radius: 4px;" src="{}"/>
</div>
'''.format(model_name, prediction_image)),
                    ui.link(
                        name=f'download_prediction_{i}',
                        label='Download Prediction',
                        path=prediction_image,
                        download=True,
                        button=True,
                    ),
                ],
                justify='center', align='center', direction='column'
            )
            )
        items += [
            ui.inline(
                items=imgs_markups,
                justify='center'
            ),
            ui.inline(
                items=[ui.text(f'''Output(s) generated by the deep learning model(s) for the file **{q.client.file_name}**- image reconstructed with high fidelity, preserving the important features.''', name='output_text'),],
                justify='center'
            )
        ]
        return items


@on('demo_file')
async def demo_file(q: Q):
    q.client.file_name = q.args.demo_file
    q.client.file_path = f"data/{q.client.file_name}"
    try:
        await download_data_from_s3(q.client.file_name)
        q.page['stepper_infos'].demo_upload.disabled = False
        await q.page.save()
    except Exception as e:
        logger.error(f"Error downloading demo file {q.client.file_name}: {str(e)}")
        q.page['meta'].dialog = ui.dialog(
            title='Error',
            closable=True,
            blocking=True,
            width='min(75%, 600px);',
            items=[
                ui.text(f"Error downloading demo file {q.client.file_name}: {str(e)}"),
            ],
        )
        await q.page.save()


@on('demo_upload')
async def demo_upload(q: Q):
    clear_cards(q)
    q.client.stepper = 1
    items = await stepper_items(q)
    add_card(q, 'stepper_infos',
            ui.form_card(
                box=ui.box('horizontal', height='900px', width='100%'),
                title='',
                items=items
                ))

@on('predict')
async def predict(q: Q):
    await loading_prediction(q)
    await predict_models(q)
    q.page['meta'].dialog = None
    clear_cards(q)
    q.client.stepper = 2
    items = await stepper_items(q)
    add_card(q, 'stepper_infos',
            ui.form_card(
                box=ui.box('horizontal', height='900px', width='100%'),
                title='',
                items=items
                ))


@on('#page3')
async def page3(q: Q):
    q.page['sidebar'].value = '#page3'
    clear_cards(q, ['form'])
    add_card(q, 'form',
    ui.form_card(box='vertical', items=[
        ui.table(
            name='table',
            columns=[
                ui.table_column(name='name', label='Name'),
                ui.table_column(name='surname', label='Surname'),
                ui.table_column(name='gtid', label='GTID'),
                ui.table_column(name='email', label='Email'),
            ], 
            rows=[
                ui.table_row(name='row1', cells=['Anderson', 'Baraldo', 'abaraldo3', 'abaraldo3@gatech.edu']),
                ui.table_row(name='row2', cells=['Luiz', 'Santos', 'lsantos49', 'lsantos49@gatech.edu']),
            ]
        ),
    ]))


models_names = {
    "unet_baseline_l1_ssim_l1_ssim": "Baseline U-Net",
    "unet_baseline_l1_ssim_roi_l1_ssim_roi": "Baseline U-Net with ROI-weighted Loss",
    "unet_cbam_l1_ssim_l1_ssim_attn": "CBAM-enhanced U-Net",
    "unet_cbam_l1_ssim_roi_l1_ssim_roi_attn": "CBAM-enhanced U-Net with ROI-weighted Loss",
    "unet_attention_gates_l1_ssim_l1_ssim_agate": "Attention Gates U-Net",
    "unet_attention_gates_l1_ssim_roi_l1_ssim_roi_agate": "Attention Gates U-Net with ROI-weighted Loss",
    "unet_full_attention_l1_ssim_l1_ssim_attn_agate": "Full Attention U-Net",
    "unet_full_attention_l1_ssim_roi_l1_ssim_roi_attn_agate": "Full Attention U-Net with ROI-weighted Loss",
}


h5_files = [
    'file1001650.h5',
    'file1002549.h5',
    'file1001499.h5',
    'file1000178.h5',
    'file1000126.h5',
    'file1000254.h5',
    'file1000314.h5',
    'file1000350.h5',
    'file1002002.h5',
    'file1000073.h5'
]

# HELPER FUNCTIONS --------------------------------------------------------------------

async def loading_prediction(q):
    q.page['meta'].dialog = ui.dialog(
        title='',
        closable=False,
        blocking=True,
        width='min(75%, 600px);',
        items=[
            ui.inline(
                items=[
                    ui.markup('<center><img  style="width: min(100%, 580px);" src="{}"/></center>'.format(q.app.loader))
                ],
                justify='center', align='center', direction='column'
            ),
        ],
    )
    await q.page.save()