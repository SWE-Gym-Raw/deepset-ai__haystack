# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging

import pytest

from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import ByteStream
from haystack.components.converters.pdfminer import PDFMinerToDocument


class TestPDFMinerToDocument:
    def test_run(self, test_files_path):
        """
        Test if the component runs correctly.
        """
        converter = PDFMinerToDocument()
        sources = [test_files_path / "pdf" / "sample_pdf_1.pdf"]
        results = converter.run(sources=sources)
        docs = results["documents"]

        assert len(docs) == 1
        for doc in docs:
            assert "the page 3 is empty" in doc.content
            assert "Page 4 of Sample PDF" in doc.content

    def test_init_params_custom(self, test_files_path):
        """
        Test if init arguments are passed successfully to PDFMinerToDocument layout parameters
        """
        converter = PDFMinerToDocument(char_margin=0.5, all_texts=True, store_full_path=False)
        assert converter.layout_params.char_margin == 0.5
        assert converter.layout_params.all_texts is True
        assert converter.store_full_path is False

    def test_run_with_store_full_path_false(self, test_files_path):
        """
        Test if the component runs correctly with store_full_path=False
        """
        converter = PDFMinerToDocument(store_full_path=False)
        sources = [test_files_path / "pdf" / "sample_pdf_1.pdf"]
        results = converter.run(sources=sources)
        docs = results["documents"]

        assert len(docs) == 1
        for doc in docs:
            assert "the page 3 is empty" in doc.content
            assert "Page 4 of Sample PDF" in doc.content
            assert doc.meta["file_path"] == "sample_pdf_1.pdf"

    def test_run_wrong_file_type(self, test_files_path, caplog):
        """
        Test if the component runs correctly when an input file is not of the expected type.
        """
        sources = [test_files_path / "audio" / "answer.wav"]
        converter = PDFMinerToDocument()

        with caplog.at_level(logging.WARNING):
            output = converter.run(sources=sources)
            assert "Is this really a PDF?" in caplog.text

        docs = output["documents"]
        assert not docs

    def test_arg_is_none(self, test_files_path):
        """
        Test if the component runs correctly when an argument is None.
        """
        converter = PDFMinerToDocument(char_margin=None)
        assert converter.layout_params.char_margin is None

    def test_run_doc_metadata(self, test_files_path):
        """
        Test if the component runs correctly when metadata is supplied by the user.
        """
        converter = PDFMinerToDocument()
        sources = [test_files_path / "pdf" / "sample_pdf_2.pdf"]
        metadata = [{"file_name": "sample_pdf_2.pdf"}]
        results = converter.run(sources=sources, meta=metadata)
        docs = results["documents"]

        assert len(docs) == 1
        assert "Ward Cunningham" in docs[0].content
        assert docs[0].meta["file_name"] == "sample_pdf_2.pdf"

    def test_incorrect_meta(self, test_files_path):
        """
        Test if the component raises an error when incorrect metadata is supplied by the user.
        """
        converter = PDFMinerToDocument()
        sources = [test_files_path / "pdf" / "sample_pdf_3.pdf"]
        metadata = [{"file_name": "sample_pdf_3.pdf"}, {"file_name": "sample_pdf_2.pdf"}]
        with pytest.raises(ValueError, match="The length of the metadata list must match the number of sources."):
            converter.run(sources=sources, meta=metadata)

    def test_run_bytestream_metadata(self, test_files_path):
        """
        Test if the component runs correctly when metadata is read from the ByteStream object.
        """
        converter = PDFMinerToDocument()
        with open(test_files_path / "pdf" / "sample_pdf_2.pdf", "rb") as file:
            byte_stream = file.read()
            stream = ByteStream(byte_stream, meta={"content_type": "text/pdf", "url": "test_url"})

        results = converter.run(sources=[stream])
        docs = results["documents"]

        assert len(docs) == 1
        assert "Ward Cunningham" in docs[0].content
        assert docs[0].meta == {"content_type": "text/pdf", "url": "test_url"}

    def test_run_bytestream_doc_overlapping_metadata(self, test_files_path):
        """
        Test if the component runs correctly when metadata is read from the ByteStream object and supplied by the user.

        There is an overlap between the metadata received.

        The component should use the supplied metadata to overwrite the values if there is an overlap between the keys.
        """
        converter = PDFMinerToDocument()
        with open(test_files_path / "pdf" / "sample_pdf_2.pdf", "rb") as file:
            byte_stream = file.read()
            # ByteStream has "url" present in metadata
            stream = ByteStream(byte_stream, meta={"content_type": "text/pdf", "url": "test_url_correct"})

        # "url" supplied by the user overwrites value present in metadata
        metadata = [{"file_name": "sample_pdf_2.pdf", "url": "test_url_new"}]
        results = converter.run(sources=[stream], meta=metadata)
        docs = results["documents"]

        assert len(docs) == 1
        assert "Ward Cunningham" in docs[0].content
        assert docs[0].meta == {"file_name": "sample_pdf_2.pdf", "content_type": "text/pdf", "url": "test_url_new"}

    def test_run_error_handling(self, caplog):
        """
        Test if the component correctly handles errors.
        """
        sources = ["non_existing_file.pdf"]
        converter = PDFMinerToDocument()
        with caplog.at_level(logging.WARNING):
            results = converter.run(sources=sources)
            assert "Could not read non_existing_file.pdf" in caplog.text
            assert results["documents"] == []

    def test_run_empty_document(self, caplog, test_files_path):
        sources = [test_files_path / "pdf" / "non_text_searchable.pdf"]
        converter = PDFMinerToDocument()
        with caplog.at_level(logging.WARNING):
            results = converter.run(sources=sources)
            assert "PDFMinerToDocument could not extract text from the file" in caplog.text
            assert results["documents"][0].content == ""

    def test_run_split_passage(self, test_files_path):
        converter = PDFMinerToDocument()
        sources = [test_files_path / "pdf" / "sample_pdf_2.pdf"]
        pdf_doc = converter.run(sources=sources)
        splitter = DocumentSplitter(split_length=1, split_by="passage")
        docs = splitter.run(pdf_doc["documents"])

        assert len(docs["documents"]) == 22
        assert docs["documents"][4].content == (
            """A wiki (/ˈwɪki/ (About this soundlisten) WIK-ee) is a hypertext publication collaboratively \nedited and
            managed by its own audience directly using a web browser. A typical wiki \ncontains multiple pages for the
            subjects or scope of the project and may be either open \nto the public or limited to use within an
            organization for maintaining its internal knowledge \nbase. Wikis are enabled by wiki software, otherwise
            known as wiki engines. A wiki engine, \nbeing a form of a content management system, diﬀers from other
            web-based systems \nsuch as blog software, in that the content is created without any deﬁned owner or
            leader, \nand wikis have little inherent structure, allowing structure to emerge according to the \nneeds
            of the users.[1] \n\n"""
        )
        assert (
            docs["documents"][5].content
            == """\nWiki engines usually allow content to be written using a simpliﬁed markup language and \nsometimes
            edited with the help of a rich-text editor.[2] There are dozens of diﬀerent wiki \nengines in use, both
            standalone and part of other software, such as bug tracking \nsystems. Some wiki engines are open source,
            whereas others are proprietary. Some \npermit control over diﬀerent functions (levels of access); for
            example, editing rights may \npermit changing, adding, or removing material. Others may permit access
            without \nenforcing access control. Other rules may be imposed to organize content. The online
             \nencyclopedia project Wikipedia is the most popular wiki-based website, and is one of the \nmost widely
             viewed sites in the world, having been ranked in the top ten since 2007.[3] \nWikipedia is not a single
             wiki but rather a collection of hundreds of wikis, with each one \npertaining to a speciﬁc language. In
             addition to Wikipedia, there are hundreds of \nthousands of other wikis in use, both public and private,
             including wikis functioning as \nknowledge management resources, notetaking tools, community websites, and
             intranets. \nThe English-language Wikipedia has the largest collection of articles: as of February 2020,
             \nit has over 6 million articles. \n\n"""
        )
