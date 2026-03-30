from click.testing import CliRunner

import llm
from llm.cli import cli


def test_mrchatterbox_path():
    runner = CliRunner()
    result = runner.invoke(cli, ["mrchatterbox", "path"])
    assert result.exit_code == 0
    expected = str(llm.user_dir() / "mrchatterbox")
    assert result.output.strip() == expected


def test_prompt():
    model = llm.get_model("mrchatterbox")
    response = model.prompt("Good day sir, what is your name?")
    text = response.text()
    assert len(text) > 10


def test_conversation():
    model = llm.get_model("mrchatterbox")
    conversation = model.conversation()
    response1 = conversation.prompt("Good day sir, what is your name?")
    text1 = response1.text()
    assert len(text1) > 10
    response2 = conversation.prompt("And what is your profession?")
    text2 = response2.text()
    assert len(text2) > 10
