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
    def test_client_side_six_tile_navigation_without_server_callbacks(self):
        code=(ROOT/"mobile_dashboard.py").read_text(encoding="utf-8")
        self.assertIn('with st.container(key="nav_grid")',code)
        self.assertIn('tabs=st.tabs(PAGES)',code)
        self.assertIn('for index,page in enumerate(PAGES):',code)
        self.assertIn('with tabs[index]:',code)
        self.assertNotIn('st.radio("Navigate"',code)
        self.assertNotIn('st.button(label,key=f"nav_{key}"',code)
        self.assertNotIn('on_click=_activate_page',code)
        self.assertIn('grid-template-columns:repeat(3,minmax(0,1fr))',code)
        self.assertIn('[aria-selected="true"]',code)

    def test_heavy_sources_only_load_on_explicit_action(self):
        research=(ROOT/"research_ui.py").read_text(encoding="utf-8")
        tracker=(ROOT/"tracker_ui.py").read_text(encoding="utf-8")
        mobile=(ROOT/"mobile_dashboard.py").read_text(encoding="utf-8")
        self.assertIn('if st.button("Load historical event seasons"',research)
        self.assertIn('if st.button("Sync cloud archive"',tracker)
        self.assertIn('if st.button("Open advanced research tools"',mobile)
        self.assertIn('for r in cards[:max_items]',mobile)

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
