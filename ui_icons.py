"""Small consistent white outline icons, inlined without emoji or external CDN."""
from urllib.parse import quote
from html import escape

PATHS={
    "brief":'<path d="m3 10 9-7 9 7v10a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1V10Z"/>',
    "star":'<path d="m12 2 3.1 6.3 7 1-5 5 .9 7-6-3.3-6 3.3.9-7-5-5 7-1L12 2Z"/>',
    "zap":'<path d="m13 2-9 12h7l-1 8 10-12h-7V2Z"/>',
    "calendar":'<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M7 3v4M17 3v4M3 10h18"/>',
    "tracker":'<path d="M4 19h16M6 15l4-4 3 3 5-7"/><path d="M15 7h3v3"/>',
    "research":'<path d="M9 3h6M10 3v6l-5.5 9A2 2 0 0 0 6 21h12a2 2 0 0 0 1.5-3L14 9V3"/><path d="M8 15h8"/>',
    "settings":'<path d="M12 15.2a3.2 3.2 0 1 0 0-6.4 3.2 3.2 0 0 0 0 6.4Z"/><path d="M19.4 15.3l1.5 1.2-1.6 2.7-1.8-.6a8.1 8.1 0 0 1-2.2 1.3l-.3 1.9h-3.1l-.3-1.9a8.1 8.1 0 0 1-2.2-1.3l-1.8.6-1.6-2.7 1.5-1.2a7.8 7.8 0 0 1 0-2.6L6 11.5l1.6-2.7 1.8.6a8.1 8.1 0 0 1 2.2-1.3l.3-1.9h3.1l.3 1.9a8.1 8.1 0 0 1 2.2 1.3l1.8-.6 1.6 2.7-1.5 1.2a7.8 7.8 0 0 1 0 2.6Z" transform="translate(-.5 -2) scale(1.03)"/>',
    "download":'<path d="M12 3v13m-4-4 4 4 4-4M4 18v3h16v-3"/>',
    "refresh":'<path d="M20 11a8 8 0 0 0-14-5L4 8M4 4v4h4M4 13a8 8 0 0 0 14 5l2-2M20 20v-4h-4"/>',
    "football":'<circle cx="12" cy="12" r="9.5"/><path d="m12 8 4 3-1.5 5h-5L8 11l4-3ZM12 8V4M8 11l-4-1M16 11l4-1M9.5 16l-2 4M14.5 16l2 4"/>',
    "shield":'<path d="m12 2 8 4v6c0 5-3.5 8-8 10-4.5-2-8-5-8-10V6l8-4Z"/><path d="m8.5 12 2.5 2.5 4.5-5"/>',
    "clock":'<circle cx="12" cy="12" r="9"/><path d="M12 7v5l4 2"/>',
}
def svg(name,size=19,color="#f6fbff"):
    paths=PATHS[name]
    return (f'<svg width="{int(size)}" height="{int(size)}" viewBox="0 0 24 24" fill="none" '
       f'stroke="{escape(color)}" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round" '
       f'aria-hidden="true">{paths}</svg>')

def css_mask(name):
    plain=(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
           f'stroke="white" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">'
           f'{PATHS[name]}</svg>')
    return "url('data:image/svg+xml,"+quote(plain,safe="")+"')"

def navigation_css():
    """Each tile is a real Streamlit button with a CSS-masked white icon.

    Unlike styling st.radio internals, this never exposes native radio circles
    or an unwanted widget title as a fake navigation tile.
    """
    icons=(("brief","brief"),("strong","star"),("upset","zap"),
           ("matches","calendar"),("tracker","tracker"),("research","research"))
    style=[]
    for key,name in icons:
        mask=css_mask(name)
        style.append(f"""
        .st-key-nav_{key} button::before {{
          content:"";display:block!important;width:22px;height:22px;
          flex:0 0 22px;background:#f5f9ff!important;
          -webkit-mask:{mask} center / contain no-repeat;
          mask:{mask} center / contain no-repeat;
        }}
        """)
    return "".join(style)
