from .controller import control_loop

class RCA_Agent:
    def __init__(self, agent_prompt, basic_prompt, task_context=None) -> None:

        self.ap = agent_prompt
        self.bp = basic_prompt
        self.task_context = task_context or {}

    def run(self, instruction, logger, max_step=25, max_turn=5, final_token_threshold=None):
            
        logger.info(f"Objective: {instruction}")
        prediction, trajectory, prompt = control_loop(
            instruction,
            "",
            self.ap,
            self.bp,
            logger=logger,
            max_step=max_step,
            max_turn=max_turn,
            task_context=self.task_context,
            max_input_tokens_before_final=final_token_threshold,
        )
        logger.info(f"Result: {prediction}")

        return prediction, trajectory, prompt
