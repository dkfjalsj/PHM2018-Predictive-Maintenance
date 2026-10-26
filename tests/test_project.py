import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from html.parser import HTMLParser
import xml.etree.ElementTree as ET
from src.config import PROJECT_ROOT, data_root, output_dir, report_workdir, read_env

class Links(HTMLParser):
    def __init__(self):super().__init__();self.refs=[];self.svg=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag in {'a','img'}:
            value=a.get('href') or a.get('src')
            if value and not value.startswith(('http:','https:','#')):self.refs.append(value)
        if tag=='svg':self.svg+=1

class ProjectTests(unittest.TestCase):
    def test_real_input_structure(self):
        self.assertTrue((data_root()/'train/train_faults').is_dir())

    def test_environment_overrides_file(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'train/train_faults').mkdir(parents=True);(p/'train/train_ttf').mkdir()
            with patch.dict(os.environ,{'PHM2018_DATA_ROOT':temp}):self.assertEqual(data_root(),p.resolve())

    def test_no_overwrite_output(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(os.environ,{'PHM2018_OUTPUT_ROOT':temp,'PHM2018_RUN_ID':'fixed'}):
                output_dir('test')
                with self.assertRaises(FileExistsError):output_dir('test')

    def test_protected_reports(self):
        for folder in ['reports/sensor_validation','reports/archive','reports/archive/sensor_validation']:
            with patch.dict(os.environ,{'PHM2018_SENSOR_REPORT_DIR':folder}):
                with self.assertRaises(ValueError):report_workdir()

    def test_reports_and_graphs(self):
        for rel in ['reports/fault_analysis/report.html','reports/sensor_validation/sensor_validation_report.html']:
            p=PROJECT_ROOT/rel;parser=Links();parser.feed(p.read_text(encoding='utf-8'))
            for ref in parser.refs:self.assertTrue((p.parent/ref).is_file(),ref)
        for p in (PROJECT_ROOT/'figures').rglob('*.svg'):ET.parse(p)

    def test_environment_template_has_no_local_path(self):
        self.assertEqual(read_env(PROJECT_ROOT/'.env.example').get('PHM2018_DATA_ROOT'),'')

if __name__=='__main__':unittest.main()
