from transformers import BartForConditionalGeneration, BartTokenizer

class BartSummariser:
    def __init__(self, text):
        self.text = text
        self.model_name = "facebook/bart-large-cnn"
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
    
    def bart_summarisation(self):
        """Summarize the input text using the BART model."""
        inputs = self.tokenizer([self.text], max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = self.model.generate(inputs.input_ids, num_beams=4, length_penalty=2.0, max_length=250, min_length=150, no_repeat_ngram_size=3)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

if __name__ == "__main__":
    text = """
Russell Brand is being investigated by a second police force in the wake of allegations about the comedian.
Thames Valley Police said a woman contacted the force two weeks ago with "new information" in relation to reports of "harassment and stalking".
Metropolitan Police previously confirmed it had received a "number of allegations of sexual offences".
Brand has been accused of rape and sexual assaults during a seven-year period at the height of his fame.
The allegations were made in a joint investigation by the Sunday Times, the Times and Channel 4's Dispatches.
The BBC understands the woman reported her allegations to Thames Valley Police numerous times between 2018 and 2022 but no further action was taken.
Mr Brand had also accused the woman of harassment against him in 2017.
'Absolutely refutes'
The force confirmed it was looking into the new information but "it would be inappropriate to comment on an ongoing investigation".
The BBC has approached Brand for a response to these claims.
The comedian and actor has previously denied "very serious criminal allegations" and "extremely egregious and aggressive attacks", which he said he "absolutely refutes".
The Dispatches programme, Russell Brand - In Plain Sight, heard four women accuse Brand of sexual assaults between 2006 and 2013.
During that time, Brand held several jobs, including at Channel 4 and BBC Radio 2.
The investigation, which aired on 16 September, claimed he had also displayed predatory and controlling behaviour, and behaved inappropriately at work.
Brand posted a video online refuting the allegations before they were aired
The 48-year-old said his relationships had "always" been consensual.
In a response to the allegations of "non-recent" sexual offences reported to the Met in September, Brand live-streamed a video on Rumble.
The actor and comedian was critical of the mainstream media but did not directly address the claims against him.
He said there was an "apparent concerted effort between the legacy media and the state to silence independent voices".
In the Sunday Times, Times and Channel 4 investigation, four women levelled accusations against Brand between 2006 and 2013.
These are the allegations against Brand:
One woman alleges he raped her without a condom against a wall in his Los Angeles home. She says Brand tried to stop her leaving until she told him she was going to the bathroom. She was reportedly treated at a rape crisis centre on the same day, which the Times says it has confirmed via medical records
A second woman, in the UK, alleges Brand assaulted her when he was in his early 30s and she was 16 and still at school. She alleges he referred to her as "the child" during an emotionally abusive and controlling relationship. Looking back, she says he "engaged in the behaviours of a groomer"
A third woman claims Brand sexually assaulted her while she worked with him in Los Angeles. She alleges she repeatedly told Brand to get off her, and when he eventually relented he "flipped" and was "super angry". She says he threatened to take legal action if she told anyone else about her allegation
A fourth woman alleges being sexually assaulted by Brand in the UK and him being physically and emotionally abusive towards her
On the same day the Dispatches allegations emerged, Brand performed a comedy gig at the Troubadour Wembley Park Theatre in north-west London, in which he alluded to the claims but did not address them directly.
He told the audience there were things he wanted to talk about but could not.
On the following Monday, the remaining dates for his Bipolarisation tour were postponed.
"""
    
    summariser = BartSummariser(text)
    summary = summariser.bart_summarisation()
    print("\n\nSummarised Text:")
    print(summary)