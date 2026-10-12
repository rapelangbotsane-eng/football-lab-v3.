import unittest
from ui_icons import svg,css_mask,navigation_css,PATHS

class WhiteIconsTests(unittest.TestCase):
    def test_all_dashboard_icons_have_inline_svg(self):
        for name in ("brief","star","zap","calendar","tracker","research","football","settings"):
            markup=svg(name)
            self.assertIn("<svg",markup)
            self.assertIn('stroke="#f6fbff"',markup)
            self.assertIn("</svg>",markup)
    def test_nav_has_six_pictograms(self):
        css=navigation_css()
        self.assertEqual(css.count("::before"),6)
        self.assertIn("data:image/svg+xml",css)
    def test_mask_contains_no_remote_asset(self):
        for icon in ("brief","research","tracker"):
            self.assertNotIn("https://",css_mask(icon))
            self.assertNotIn("http://",css_mask(icon))

if __name__=="__main__":unittest.main()
