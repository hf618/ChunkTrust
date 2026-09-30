import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import pi05_four_queue as queue

class FocusQueue(unittest.TestCase):
    tasks=['place_a2b_left','place_bread_basket','place_bread_skillet','place_can_basket']
    def run_case(self,missing=0,adopt=False):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);events=[]
            c=dict(queue_id='test',tasks=self.tasks,easy_root=str(root/'easy'))
            record=root/'adopt.json';expected={'start_ticks':'123','command':'python panel_run_block.py --backbone pi05 --task place_can_basket'}
            record.write_text(json.dumps(dict(child_pid=777,child_identity=expected,command=['same child'],started_at=1)))
            def block(stage,bb,task,methods,extras):events.append(('hard',bb,task,extras))
            def launch(name,path,cpu):events.append(('easy',path,cpu))
            with patch.object(queue,'EXP',root),patch.object(queue,'config',return_value=c),patch.object(queue,'validate',return_value={}),patch.object(queue,'block',side_effect=block),patch.object(queue,'panel_jobs',return_value=[]),patch.object(queue,'verify_saved_jobs',return_value=missing),patch.object(queue,'scoped_rows',return_value={'Hard':[None]*4800,'Easy':[None]*4800}),patch.object(queue.runner,'run',side_effect=launch),patch.object(queue,'identity',side_effect=[expected,expected,None]),patch.object(queue.time,'sleep'):
                if missing:
                    with self.assertRaises(AssertionError):queue.main()
                else:queue.main(record if adopt else None)
            return events

    def test_only_four_pi05_hard_then_easy_and_adopt_does_not_launch_duplicates(self):
        for adopt in (False,True):
            events=self.run_case(adopt=adopt)
            self.assertEqual(events[:4],[('hard','pi05',t,[0,100,200]) for t in self.tasks])
            self.assertEqual(len(events),5);self.assertEqual(events[-1][0],'easy')
            self.assertTrue(events[-1][1].endswith('/runtime/easy_pipeline.py'))
            self.assertTrue(events[-1][2])

    def test_no_easy_transition_if_hard_is_incomplete(self):
        events=self.run_case(missing=1)
        self.assertEqual(events,[('hard','pi05',self.tasks[0],[0,100,200])])

if __name__=='__main__':unittest.main()
