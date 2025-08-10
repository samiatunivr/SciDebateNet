class GeneratorAgent:
    def generate(self, topic):
        return f"Hypothesis about {topic}"

class CriticAgent:
    def critique(self, hypothesis):
        return f"Critique of: {hypothesis}"

class SynthesizerAgent:
    def synthesize(self, hypo, critique):
        return f"Refined hypothesis based on: {hypo} and {critique}"
