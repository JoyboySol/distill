import unittest

from distill.core.judge import judge_output, judge_output_with_timeout


class InstructionFollowingJudgeTests(unittest.TestCase):

    def test_structured_if_constraints_pass_when_all_supported_rules_match(self):
        row_data = {
            "ground_truth": (
                "[{'instruction_id': ['detectable_format:sentence_hyphens', "
                "'last_word:last_word_answer'], 'kwargs': [None, "
                "{'last_word': 'brief'}]}]"
            ),
            "dataset": "ifeval",
        }
        messages = [{
            "role": "assistant",
            "content": (
                "Kpanlogo-is-a-percussion-instrument-while-Shamisen-is-a-"
                "string-instrument-brief"
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"], "instruction_following")
        self.assertEqual(result["judge_backend"], "instruction_following_v1")
        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_structured_if_all_unsupported_constraints_default_to_true(self):
        row_data = {
            "ground_truth": (
                "[{'instruction_id': ['count:lowercase_counting'], 'kwargs': "
                "[{'N': 3}]}]"
            ),
            "dataset": "ifeval",
        }
        messages = [{
            "role": "assistant",
            "content": "Plain response",
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"], "instruction_following")
        self.assertEqual(result["judge_backend"], "instruction_following_v1")
        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_structured_if_partially_supported_constraints_only_judge_supported_subset(self):
        row_data = {
            "ground_truth": (
                "[{'instruction_id': ['last_word:last_word_answer', "
                "'count:lowercase_counting'], 'kwargs': [{'last_word': "
                "'brief'}, {'N': 3}]}]"
            ),
            "dataset": "ifeval",
        }
        messages = [{
            "role": "assistant",
            "content": "Concise answer brief",
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"], "instruction_following")
        self.assertEqual(result["judge_backend"], "instruction_following_v1")
        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_structured_if_partially_supported_constraints_can_still_fail(self):
        row_data = {
            "ground_truth": (
                "[{'instruction_id': ['last_word:last_word_answer', "
                "'count:lowercase_counting'], 'kwargs': [{'last_word': "
                "'brief'}, {'N': 3}]}]"
            ),
            "dataset": "ifeval",
        }
        messages = [{
            "role": "assistant",
            "content": "Concise answer summary",
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"], "instruction_following")
        self.assertEqual(result["judge_backend"], "instruction_following_v1")
        self.assertFalse(result["is_correct"])
        self.assertEqual(result["judge_status"], "wrong_answer")

    def test_tulu_prompt_constraints_pass_for_excluded_words_and_specific_ending(self):
        row_data = {
            "prompt": (
                "Provide a detailed analysis of Candace Parker's defensive "
                "techniques in her recent games, excluding the words "
                "\"aggressive\" and \"blocking\". End your response with "
                "\"Go team.\""
            ),
            "constraints": [
                "keywords:exclude words",
                "specific ending",
            ],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "Her defensive timing stands out because she anticipates "
                "passing lanes and rotates early to help teammates. Go team."
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"], "instruction_following")
        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_prompt_constraints_only_judge_supported_subset(self):
        row_data = {
            "prompt": (
                "Write a 1 paragraph description of a retired hip-hop "
                "artist's last on-stage performance. Your response should be "
                "entirely in Spanish and end with the phrase "
                "\"Hasta pronto.\""
            ),
            "constraints": [
                "response language",
                "specific ending",
            ],
        }
        messages = [{
            "role": "assistant",
            "content": "A vivid farewell performance closed the night. Hasta pronto.",
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"], "instruction_following")
        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_prompt_constraints_supported_subset_can_still_fail(self):
        row_data = {
            "prompt": (
                "Write a 1 paragraph description of a retired hip-hop "
                "artist's last on-stage performance. Your response should be "
                "entirely in Spanish and end with the phrase "
                "\"Hasta pronto.\""
            ),
            "constraints": [
                "response language",
                "specific ending",
            ],
        }
        messages = [{
            "role": "assistant",
            "content": "A vivid farewell performance closed the night.",
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"], "instruction_following")
        self.assertFalse(result["is_correct"])
        self.assertEqual(result["judge_status"], "wrong_answer")

    def test_tulu_prompt_constraints_fail_when_required_keyword_is_missing(self):
        row_data = {
            "prompt": (
                "Write a fan chant for the Punjab Panthers. Include the "
                "keywords \"roar,\" \"victory,\" and \"spirit.\" Use no "
                "commas in the chant."
            ),
            "constraints": [
                "include keywords",
                "punctuation:use no comma",
            ],
        }
        messages = [{
            "role": "assistant",
            "content": "Roar for victory tonight Panthers rise high",
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"], "instruction_following")
        self.assertFalse(result["is_correct"])
        self.assertEqual(result["judge_status"], "wrong_answer")

    def test_tulu_keywords_ignore_prompt_punctuation_noise(self):
        row_data = {
            "prompt": (
                "Write a fan chant for the Punjab Panthers. Include the "
                "keywords \"roar,\" \"victory,\" and \"spirit.\" Use no "
                "commas in the chant."
            ),
            "constraints": [
                "include keywords",
                "punctuation:use no comma",
            ],
        }
        messages = [{
            "role": "assistant",
            "content": "Roar for victory and spirit tonight Panthers rise high",
        }]

        result = judge_output(row_data, messages)

        self.assertEqual(result["judge_type"], "instruction_following")
        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_include_keywords_ignores_apostrophes_inside_words(self):
        row_data = {
            "prompt": (
                "Analyze the economic policies of the Alberta New Democratic "
                "Party (NDP) and their impact on the province's oil industry. "
                "Your response should contain the keywords: 'pipeline', "
                "'tax', and 'innovation'."
            ),
            "constraints": ["include keywords"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "The province can pursue pipeline investment, balanced tax "
                "policy, and energy-sector innovation at the same time."
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_specific_ending_supports_exact_sentence_phrasing(self):
        row_data = {
            "prompt": (
                "Write a speech for a pharmaceutical company executive "
                "addressing a global conference. Conclude with the exact "
                "sentence: \"Innovation is not just a goal; it is our "
                "responsibility to the world.\""
            ),
            "constraints": ["specific ending"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "Research partnerships matter because they turn scientific "
                "breakthroughs into treatments that scale globally. "
                "Innovation is not just a goal; it is our responsibility to "
                "the world."
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_response_language_supports_explicit_english_requirement(self):
        row_data = {
            "prompt": (
                "Write a 2-paragraph summary about the benefits of protein in "
                "muscle building. Your response should be in English."
            ),
            "constraints": ["response language"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "Protein supports muscle repair after training.\n\n"
                "It also helps athletes recover and adapt over time."
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_in_english_and_capital_is_supported(self):
        row_data = {
            "prompt": (
                "Create a motivational poster slogan for a young golf "
                "enthusiast. Make sure the slogan is in English and all "
                "capital letters."
            ),
            "constraints": ["in english and capital"],
        }
        messages = [{
            "role": "assistant",
            "content": "CHASE EVERY SWING WITH DISCIPLINE AND HEART",
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_english_lowercase_is_supported(self):
        row_data = {
            "prompt": (
                "Write two bullet lists about dance innovation. The entire "
                "response must be in english and lowercase."
            ),
            "constraints": ["case:in english and lowercase"],
        }
        messages = [{
            "role": "assistant",
            "content": "- motion capture tools\n- adaptive choreography",
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_number_of_paragraphs_is_supported(self):
        row_data = {
            "prompt": (
                "Write a short essay about cycling history. The essay should "
                "be exactly two paragraphs long."
            ),
            "constraints": ["length constraints:number of paragraphs"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "Cycling grew rapidly as a competitive and recreational sport "
                "in the late nineteenth century.\n\n"
                "Teams and stage races later turned the sport into a global "
                "spectacle with distinct regional traditions."
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_number_of_sentences_is_supported(self):
        row_data = {
            "prompt": (
                "Write a summary of a race car driver's latest race "
                "performance in exactly 3 sentences."
            ),
            "constraints": ["length constraints:number of sentences"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "The driver started cleanly and gained two positions by lap "
                "five. Tire management kept the pace stable through the "
                "middle stint. A decisive final overtake secured a podium "
                "finish."
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_json_format_is_supported(self):
        row_data = {
            "prompt": (
                "Research and compile information about a historical event. "
                "Present your findings in a JSON format with each piece of "
                "information in a separate field named 'fact_1' and "
                "'fact_2'."
            ),
            "constraints": ["format:use json format"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                '{"fact_1": "The Apollo 11 mission landed on the moon in '
                '1969.", "fact_2": "Neil Armstrong was the first person to '
                'walk on the lunar surface."}'
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_number_of_sections_is_supported(self):
        row_data = {
            "prompt": (
                "Create a detailed guide for a running route. The guide "
                "should include 3 sections, each marked with Section N."
            ),
            "constraints": ["format:number of sections"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "Section 1: Start Point\nChoose a shaded trailhead.\n\n"
                "Section 2: Scenic Stretch\nFollow the riverside path.\n\n"
                "Section 3: Cooldown\nFinish near a water fountain."
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_number_of_words_supports_at_most_requirement(self):
        row_data = {
            "prompt": (
                "List some popular JavaScript libraries or frameworks for "
                "front-end development. Provide a short answer, not exceeding "
                "20 words."
            ),
            "constraints": ["length constraints:number of words"],
        }
        messages = [{
            "role": "assistant",
            "content": "React, Vue, Angular, Svelte, and Next.js remain widely used choices.",
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_bullet_points_requirement_is_supported(self):
        row_data = {
            "prompt": (
                "Compose an educational article about cultural heritage. "
                "Incorporate exactly 3 bullet points in a list to highlight "
                "the main elements."
            ),
            "constraints": ["format:number of bullet lists"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "Cultural heritage benefits from careful interpretation.\n"
                "- Historical continuity\n"
                "- Shared craftsmanship\n"
                "- Community memory"
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_placeholder_count_is_supported(self):
        row_data = {
            "prompt": (
                "Create a fieldwork template with at least 3 placeholders "
                "for customizable observations such as [species]."
            ),
            "constraints": ["content:number of placeholders"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "- Location: [site]\n"
                "- Species: [species]\n"
                "- Notes: [observation]"
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_postscript_is_supported(self):
        row_data = {
            "prompt": (
                "Please draft a motivational email for my team. Include a "
                "postscript with a quote from a famous basketball player."
            ),
            "constraints": ["content:include a postscript"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "Team, stay disciplined and trust the preparation.\n\n"
                "P.S. \"Excellence is the gradual result of always striving "
                "to do better.\""
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_choose_one_from_options_is_supported(self):
        row_data = {
            "prompt": (
                "Choose from the following options: ('500', '1,000', "
                "'1,500') and present your findings in a bullet list."
            ),
            "constraints": ["format:choose one from options"],
        }
        messages = [{
            "role": "assistant",
            "content": "- 1,000",
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_capital_word_frequency_is_supported(self):
        row_data = {
            "prompt": (
                "Write a fan chant for the Punjab Panthers. Make sure that "
                "the number of words in all capital letters is exactly 4."
            ),
            "constraints": ["case: frequency of capital words"],
        }
        messages = [{
            "role": "assistant",
            "content": "ROAR WITH TRUE PANTHER spirit tonight",
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_first_word_of_nth_paragraph_is_supported(self):
        row_data = {
            "prompt": (
                "Draft a 4 paragraph letter to the local government. Ensure "
                "that the first word of the third paragraph is "
                "\"Additionally\"."
            ),
            "constraints": ["length constraints:first word of the nth paragraph"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "Funding protects public collections and archives.\n\n"
                "Renovation would improve visitor access and safety.\n\n"
                "Additionally more storage space would help preserve fragile "
                "artifacts.\n\n"
                "Thank you for considering this request."
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_keyword_frequency_is_supported(self):
        row_data = {
            "prompt": (
                "Write a speech advocating for increased funding for law "
                "enforcement that includes the word \"safety\" at least four "
                "times and the word \"justice\" at least three times."
            ),
            "constraints": ["keywords:frequency"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "Safety guides our neighborhoods, safety strengthens trust, "
                "and safety keeps families confident. Justice must be fair, "
                "justice must be visible, and justice must protect the "
                "vulnerable. Safety and justice together create lasting "
                "stability."
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_highlighted_sections_is_supported(self):
        row_data = {
            "prompt": (
                "Write your response as if you are speaking directly to your "
                "fans. Include two highlighted sections within your response."
            ),
            "constraints": ["format:number of highlighted sections"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "I am grateful for your support through this difficult time. "
                "**Protect the music** and **stand with artists** so that "
                "creative work can keep reaching people honestly."
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_letter_frequency_is_supported(self):
        row_data = {
            "prompt": (
                "Describe in 3 sentences a memorable race. Each sentence "
                "must contain the letter 'r' at least 5 times."
            ),
            "constraints": ["keywords:letter frequency"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "Rory raced around the roaring circuit with fierce courage "
                "and vibrant fervor. Drivers roared through sharp corners "
                "while the crowd cheered for their favorite racer. After the "
                "sprint, reporters recorded every thrilling turn and daring "
                "recovery."
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_choose_one_from_following_phrase_is_supported(self):
        row_data = {
            "prompt": (
                "What percentage of people believe that vaccines cause "
                "autism? Choose from the following: ('5%', '10%', '15%') "
                "-- please include the exact phrase in your response."
            ),
            "constraints": ["format:choose one from options"],
        }
        messages = [{
            "role": "assistant",
            "content": "10%",
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_capitalized_words_per_paragraph_is_supported(self):
        row_data = {
            "prompt": (
                "The analysis should be at least three paragraphs long. "
                "Additionally, there should be no more than three capitalized "
                "words in each paragraph."
            ),
            "constraints": ["case: frequency of capital words"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "This review stays measured while noting ONE sharp contrast.\n\n"
                "Another paragraph keeps focus on TWO recurring patterns.\n\n"
                "The final paragraph mentions THREE moments without excess."
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_nth_paragraph_starts_with_word_is_supported(self):
        row_data = {
            "prompt": (
                "As a space analyst, draft a 4-paragraph report. Ensure the "
                "third paragraph starts with the word \"Furthermore\"."
            ),
            "constraints": ["length constraints:first word of the nth paragraph"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "China's lunar program has expanded steadily.\n\n"
                "International responses remain mixed.\n\n"
                "Furthermore new mission timelines could shift collaboration "
                "priorities.\n\n"
                "Upcoming launches will test that momentum."
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_bold_text_sections_requirement_is_supported(self):
        row_data = {
            "prompt": (
                "Please ensure there are at least *two bold text sections* "
                "in your response."
            ),
            "constraints": ["format:number of highlighted sections"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "Use **clear hooks** to open strongly and **repeat core "
                "benefits** so the message stays memorable."
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_letter_frequency_word_repetition_variant_is_supported(self):
        row_data = {
            "prompt": (
                "Tell me about the life and career of Twila Paris in 3 "
                "sentences. Each sentence must include the word 'gospel' at "
                "least twice."
            ),
            "constraints": ["keywords:letter frequency"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "Her gospel songwriting helped shape contemporary gospel "
                "music. Twila Paris brought gospel themes into gospel "
                "recordings with unusual clarity. Later, gospel audiences "
                "praised her gospel catalog for its consistency."
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_global_letter_frequency_is_supported(self):
        row_data = {
            "prompt": (
                "Write a warm note to a community mentor. Ensure that the "
                "letter 'e' appears at least 8 times."
            ),
            "constraints": ["keywords:letter frequency"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "Dear mentor, we deeply appreciate the steady presence and "
                "generous care you keep offering everyone here."
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_word_range_requirement_is_supported(self):
        row_data = {
            "prompt": (
                "Summarize Bayesian analysis in a short line. Your response "
                "should be between 5 and 7 words."
            ),
            "constraints": ["length constraints:number of words"],
        }
        messages = [{
            "role": "assistant",
            "content": "Bayesian models improve modern trial analysis",
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_hyphenated_word_count_requirement_is_supported(self):
        row_data = {
            "prompt": (
                "Write a 6-word slogan for a neighborhood library."
            ),
            "constraints": ["length constraints:number of words"],
        }
        messages = [{
            "role": "assistant",
            "content": "Read together, learn daily, grow endlessly",
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_highlighted_sections_contain_variant_is_supported(self):
        row_data = {
            "prompt": (
                "List three bakery items for a music-night menu. Your answer "
                "should contain two highlighted sections."
            ),
            "constraints": ["format:number of highlighted sections"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "**Berry tart** pairs well with coffee, while **brioche "
                "buns** work for savory sliders."
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_choose_one_final_sentence_options_is_supported(self):
        row_data = {
            "prompt": (
                "Write a detailed report evaluating new teaching methods. "
                "Conclude the report by choosing one of the following "
                "options as your final sentence: \"Further research is "
                "needed to validate these findings.\" or \"The current "
                "evidence supports a cautious adoption of these methods.\""
            ),
            "constraints": ["format:choose one from options"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "Pilot classrooms showed stronger participation and more "
                "consistent reflection. The current evidence supports a "
                "cautious adoption of these methods."
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_placeholder_include_variant_is_supported(self):
        row_data = {
            "prompt": (
                "Create a short story about an old television star. The "
                "story should include two placeholders where a show name and "
                "a movie title can be inserted."
            ),
            "constraints": ["content:number of placeholders"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "She still smiled when fans mentioned [show title], and she "
                "laughed whenever [movie title] came up in interviews."
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_tulu_specific_ending_phrase_variant_is_supported(self):
        row_data = {
            "prompt": (
                "Create a guide for new Terraform users and end with the "
                "phrase \"happy provisioning!\"."
            ),
            "constraints": ["specific ending"],
        }
        messages = [{
            "role": "assistant",
            "content": (
                "Start with a small plan, review every variable, and test "
                "changes in a safe environment before scaling out. happy "
                "provisioning!"
            ),
        }]

        result = judge_output(row_data, messages)

        self.assertTrue(result["is_correct"])
        self.assertEqual(result["judge_status"], "pass")

    def test_timeout_wrapper_preserves_instruction_following_result(self):
        row_data = {
            "ground_truth": (
                "[{'instruction_id': ['last_word:last_word_answer'], "
                "'kwargs': [{'last_word': 'brief'}]}]"
            ),
            "dataset": "ifeval",
        }
        messages = [{
            "role": "assistant",
            "content": "Concise answer brief",
        }]

        direct = judge_output(row_data, messages)
        guarded = judge_output_with_timeout(row_data, messages, timeout=20)

        self.assertEqual(guarded, direct)


if __name__ == "__main__":
    unittest.main()
