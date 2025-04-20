from h2o_wave import Q, ui
from typing import Optional, List


def add_card(q, name, card) -> None:
    q.client.cards.add(name)
    q.page[name] = card


def clear_cards(q, ignore: Optional[List[str]] = []) -> None:
    if not q.client.cards:
        return

    for name in q.client.cards.copy():
        if name not in ignore:
            del q.page[name]
            q.client.cards.remove(name)


async def init(q: Q) -> None:
    q.page['meta'] = ui.meta_card(
        box='',
        animate=True,
        stylesheet=ui.inline_stylesheet(get_style()),
        layouts=[ui.layout(breakpoint='xs', min_height='100vh', zones=[
            ui.zone('main', size='1', direction=ui.ZoneDirection.ROW, zones=[
                ui.zone('sidebar', size='250px'),
                ui.zone('body', zones=[
                    ui.zone('header'),
                    ui.zone('content', zones=[
                        ui.zone('horizontal', direction=ui.ZoneDirection.ROW),
                        ui.zone('vertical'),
                        ui.zone('grid', direction=ui.ZoneDirection.ROW, wrap='stretch', justify='center')
                    ]),
                ]),
            ])
        ])]
    )
    image, = await q.site.upload(['static/icon.png'])
    q.page['sidebar'] = ui.nav_card(
        box='sidebar',
        color='primary',
        title='MRI Reconstruction Acceleration',
        subtitle="An approach using Deep Learning",
        value=f'#{q.args["#"]}' if q.args['#'] else '#page1',
        image=image,
        items=[
            ui.nav_group('Menu', items=[
                ui.nav_item(name='#page1', label='Home', icon='Home'),
                ui.nav_item(name='#page2', label='Model Prediction', icon='IOT'),
                ui.nav_item(name='#page3', label='Group Members', icon='Group'),
            ]),
        ])
    q.page['header'] = ui.header_card(
        box='header',
        title='',
        subtitle='',
        secondary_items=[ui.text_l('**Final Project** - CS6440 Health Informatics')],
        items=[
            ui.persona(title='Group 92', subtitle='CS6440', size='xs', initials='G24'),
        ]
    )



def get_style():
    style = """
div[data-test="vertical"] > div{
    margin-top: 0px;
    margin-bottom: 0px;
}
div[data-test="stepper_prediction"] > div{
    margin-bottom: 50px;
}
button[data-test="predict"]{
    border-radius: 10px;
}
button[data-test^="download_prediction_"] {
    border-radius: 20px;
}
div[data-test="output_text"] > div{
    margin-top: 30px;
}
"""
    return style
    