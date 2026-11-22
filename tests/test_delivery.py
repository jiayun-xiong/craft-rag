from pathlib import Path
from unittest import TestCase
from src.generation.pipeline import Draft, Finding, validate
from src.nc.turning import Setup, contour, plan, postprocess, verify

ROOT = Path(__file__).resolve().parents[1]


class DeliveryTests(TestCase):
    def test_reference_is_bound_to_original(self):
        hit = {"text":"MC6015 FP 每转进给 0.08～0.25 mm/rev"}
        f = Finding(field="进给",value="0.08～0.25 mm/rev",ref=1)
        draft = Draft(status="answered",findings=[f],missing=[])
        self.assertEqual(validate(draft,[hit]).findings[0].quote,hit["text"])
        f.value = "0.8 mm/rev"
        with self.assertRaises(ValueError):
            validate(draft,[hit])

    def test_unknown_reference_and_empty_answer(self):
        draft=Draft(status="answered",findings=[],missing=[])
        with self.assertRaises(ValueError):
            validate(draft,[])
        draft.findings=[Finding(field="工序",value="精车",ref=2)]
        with self.assertRaises(ValueError):
            validate(draft,[{"text":"精车"}])

    def test_example_and_damaged_program(self):
        job=Setup.model_validate_json((ROOT/'examples/教学工单.json').read_text())
        sections=contour(ROOT/'examples/两阶外圆.dxf')
        limits=[(230,395),(.08,.25),(.1,1)]
        code=postprocess(plan(sections,job,limits),job)
        self.assertEqual(verify(code,sections,job),[30,31])
        with self.assertRaises(ValueError):
            verify(code.replace("G01 Z-40","G00 Z-40"),sections,job)
        with self.assertRaises(ValueError):
            plan(sections,job.model_copy(update={"free_length_mm":30}),limits)
        with self.assertRaises(ValueError):
            plan(sections,job.model_copy(update={"feed_mm_rev":.5}),limits)

    def test_program_ambiguity_requires_two_distinct_locations(self):
        from src.generation.lookup import lookup
        hits=[{"text":"O0123","source":"a.pdf","page":1},{"text":"O0123","source":"a.pdf","page":2}]
        query="程序号O0123能够唯一定位吗？"
        self.assertEqual(len(lookup(query,hits)["findings"]),2)
        self.assertIsNone(lookup(query,[hits[0],hits[0]]))
        self.assertIsNone(lookup("O0123 做了哪些加工？",hits))

    def test_three_section_profile_and_unsupported_arc(self):
        import ezdxf
        from tempfile import TemporaryDirectory

        job = Setup.model_validate_json((ROOT / 'examples/教学工单.json').read_text())
        doc = ezdxf.new(units=4)
        line = doc.modelspace().add_lwpolyline(
            [(0, 14.5), (-10, 14.5), (-10, 15), (-20, 15), (-20, 15.5), (-40, 15.5)],
            dxfattribs={'layer': 'TURN_PROFILE'},
        )
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'profile.dxf'
            doc.saveas(path)
            sections = contour(path)
            code = postprocess(plan(sections, job, [(230,395), (.08,.25), (.1,1)]), job)
            self.assertEqual(verify(code, sections, job), [29, 30, 31])
            line[0] = (0, 14.5, 0, 0, 0.5)
            doc.saveas(path)
            with self.assertRaisesRegex(ValueError, '直线'):
                contour(path)

    def test_failed_requests_do_not_meet_latency_target(self):
        import contextlib
        import io
        import json
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from evaluate_system import run

        with TemporaryDirectory() as temp, patch('evaluate_system.ask', side_effect=ConnectionError):
            dest = Path(temp) / 'offline'
            with contextlib.redirect_stdout(io.StringIO()):
                run(dest)
            stats = json.loads((dest / 'metrics.json').read_text())
            self.assertEqual(stats['errors'], stats['n'])
            self.assertEqual(stats['within15'], 0)
            self.assertIsNone(stats['p95_s'])
