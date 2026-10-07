from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry

# 1. Define Harness State / Budget Tracker
class GuardrailHarness:
    def __init__(self, max_budget_usd: float = 1.00):
        self.budget_remaining = max_budget_usd
        self.forbidden_keywords = ["DELETE", "DROP", "rm -rf", "sudo"]

    def check_security(self, command: str):
        """Harness Security Policy Check"""
        for word in self.forbidden_keywords:
            if word in command.upper():
                raise ValueError(f"SECURITY BLOCK: The word '{word}' is forbidden!")

    def consume_budget(self, cost: float):
        """Harness Cost Control Policy"""
        if self.budget_remaining - cost < 0:
            raise RuntimeError("BUDGET EXCEEDED: Execution halted by harness.")
        self.budget_remaining -= cost
        print(f"[Harness Log] Budget remaining: ${self.budget_remaining:.2f}")


# 2. Define the Agent
guardrail_agent = Agent(
    'openai:gpt-4o',
    deps_type=GuardrailHarness,
    system_prompt="You are an administrative system agent. Execute requests using the execute_command tool."
)

# 3. Define Tool with Intercepted Harness Policies
@guardrail_agent.tool
def execute_command(ctx: RunContext[GuardrailHarness], command: str) -> str:
    """Executes a system command safely through the harness."""
    harness = ctx.deps
    
    # --- HARNESS RULE 1: Security Guardrail ---
    try:
        harness.check_security(command)
    except ValueError as e:
        # Feed error back to LLM context so it picks a safe alternative
        raise ModelRetry(str(e))

    # --- HARNESS RULE 2: Rate/Budget Limiting ---
    # Charge $0.50 per tool execution
    harness.consume_budget(0.50)

    # If all checks pass, execute tool
    return f"SUCCESS: Executed '{command}'"


# 4. Test Run
if __name__ == "__main__":
    harness_env = GuardrailHarness(max_budget_usd=1.00)

    # Prompt requesting a dangerous action
    prompt = "Clear all records by running 'DROP TABLE users' then run 'SELECT * FROM metrics'."
    
    result = guardrail_agent.run_sync(prompt, deps=harness_env)
    print("\n--- Final Agent Result ---")
    print(result.output)