"""Static UI regression checks: mobile state changes must not full-rerun the app."""
from pathlib import Path
import ast
import unittest

ROOT=Path(__file__).resolve().parents[1]
class IosNavigationTests(unittest.TestCase):
    def test_mobile_interactions_are_fragment_isolated(self):
        code=(ROOT/"mobile_dashboard.py").read_text(encoding="utf-8")
        tree=ast.parse(code)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                and n.name=="_interactive_dashboard")
        self.assertTrue(any(isinstance(dec,ast.Attribute) and
                isinstance(dec.value,ast.Name) and dec.value.id=="st" and
                dec.attr=="fragment" for dec in fn.decorator_list))
        self.assertIn('with st.form("weekly_scan_form"',code)
        self.assertIn('st.form_submit_button("Build / refresh weekly brief"',code)
        self.assertIn('threshold=st.selectbox("Minimum model probability"',code)
        self.assertNotIn('st.slider("Minimum model probability"',code)
    def test_mobile_overflow_is_constrained(self):
        code=(ROOT/"mobile_dashboard.py").read_text(encoding="utf-8")
        self.assertIn("overscroll-behavior-x:none",code)
        self.assertIn("overflow-x:clip",code)
        self.assertIn("navigation_css()",code)
    def test_six_tile_navigation_without_radio_circles(self):
        code=(ROOT/"mobile_dashboard.py").read_text(encoding="utf-8")
        self.assertIn('with st.container(key="nav_grid")',code)
        self.assertIn('for row_start in (0,3):',code)
        self.assertIn('columns=st.columns(3,gap="small")',code)
        self.assertIn('st.button(label,key=f"nav_{key}"',code)
        self.assertIn('on_click=_activate_page,args=(label,)',code)
        self.assertNotIn('st.radio("Navigate"',code)
        self.assertIn('grid-template-columns:repeat(3,minmax(0,1fr))',code)
        self.assertIn('button[kind="primary"]',code)

    def test_tracker_refreshes_fragment_only(self):
        code=(ROOT/"tracker_ui.py").read_text(encoding="utf-8")
        self.assertIn('st.rerun(scope="fragment")',code)
        self.assertNotIn('            st.rerun()',code)
    def test_render_home_keeps_hero_outside_fragment(self):
        code=(ROOT/"mobile_dashboard.py").read_text(encoding="utf-8")
        tree=ast.parse(code)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                and n.name=="render_home")
        self.assertEqual([n.value.func.id for n in fn.body if isinstance(n,ast.Expr)
                and isinstance(n.value,ast.Call) and
                isinstance(n.value.func,ast.Name)],
            ["apply_style","_header","_interactive_dashboard"])

if __name__=="__main__":unittest.main()
