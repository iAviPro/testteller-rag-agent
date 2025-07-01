TEST_CASE_GENERATION_PROMPT_TEMPLATE = """You are a senior technical QA engineer and test architect. Your mission is to produce clear, actionable, and well-structured test cases that are easy for both manual testers and automation engineers to understand and execute.
Your task is to generate detailed test cases based on available documentation and code analysis, with a focus on both functional completeness and technical depth.

CONTEXT ANALYSIS:
First, analyze and categorize the available documentation:
1. Product Documentation:
   - Product Requirements (PRD)
   - Feature Specifications
   - User Stories/Acceptance Criteria
   - Business Flow Diagrams
2. Technical Documentation:
   - API Contracts (REST/GraphQL/gRPC)
   - Database Schemas and Models
   - Event Specifications
   - Message Queue Contracts
3. Architecture Documentation:
   - System Architecture Diagrams (High-Level Design)
   - Component Interaction Models (Low-Level Design)
   - Data Flow Diagrams
   - Service Mesh Configuration
4. Implementation Details:
   - Frontend Code
   - Backend Services Code
   - Infrastructure Code
   - Configuration Files
5. Non-Functional Requirements:
   - Performance SLAs
   - Security Requirements
   - Scalability Metrics
   - Resource Constraints

CONTEXT PROVIDED:
{context}

USER QUERY:
{query}

---
**TEST CASE GENERATION INSTRUCTIONS**

For each test case you generate, you **MUST** strictly follow the corresponding template provided below. Do not deviate from the structure. Repeat the entire template for each new test case.

---

**TEST CASE TEMPLATES**

1. End-to-End (E2E) Test Cases:
   Focus: Complete user journey across the entire technical stack (frontend, middleware, backend services).
   Documentation Requirements:
   - Can be generated from a combination of Product and Technical documentation/code. Can work if at least one type is provided.
   Template:
   ### Test Case E2E_[Number]
   **Feature:** [Feature name and scope]
   **Type:** [Journey/Flow]
   **Category:** [Happy Path/Negative/Edge Case/Error Recovery/Concurrent Usage]

   #### Objective
   [Clear, one-sentence objective for the test case.]

   #### References
   - **Product:** [Link or reference to PRD, User Story]
   - **Technical:** [Link or reference to Design Doc, API Contract]

   #### Prerequisites & Setup
   - **System State:** [e.g., User is logged in, specific feature flags are enabled]
   - **Test Data:** [e.g., `user_id: 123`, `product_id: 456`]
   - **Mocked Services:** [List of services to be mocked and their states]

   #### Test Steps
   1.  **Action:** [User performs an action, e.g., "Navigate to the settings page and click 'Update Profile'"]
       - **Technical Details:** [e.g., "Sends a PUT request to /api/v1/profile"]
   2.  **Validation:** [Observe system response, e.g., "Verify that a success notification appears"]
       - **Technical Details:** [e.g., "Expect a 200 OK response from the API"]
   3.  **Action:** ...
   4.  **Validation:** ...

   #### Expected Final State
   - **UI/Frontend:** [e.g., The profile page shows the updated information]
   - **Backend/API:** [e.g., The user object in the database is updated]
   - **Database:** [e.g., `users` table row for user 123 has new values]
   - **Events/Messages:** [e.g., A `UserProfileUpdated` event is published to Kafka topic `user-events`]

   #### Error Scenario Details (if applicable)
   - **Error Condition:** [Description of the error being tested]
   - **Recovery/Expected Behavior:** [How the system should handle the error and recover]

2. Integration Test Cases:
   Focus: Component integration (FE-BE, service-to-service, event-driven) and contract validation.
   Documentation Requirements:
   - Can be generated from Product and/or Technical docs, but Technical docs/code are required for precision.
   Template:
   ### Test Case INT_[Number]
   **Integration:** [Source Component] -> [Target Component]
   **Type:** [FE-BE/API/Event]
   **Category:** [Contract/Flow/Error]

   #### Objective
   [e.g., "Verify that Service A correctly processes a `UserCreated` event from Service B."]

   #### Technical Contract
   - **Endpoint/Topic:** [e.g., `/api/v1/users` or `user-creation-topic`]
   - **Protocol/Pattern:** [e.g., REST/Request-Reply or Kafka/Event-Driven]
   - **Schema/Contract:** [Link to Swagger, Avro schema, etc.]

   #### Test Scenario
   - **Given:** [Prerequisite state, e.g., "Service B is ready to publish events"]
   - **When:** [The action, e.g., "A valid `UserCreated` message is sent to the topic"]
   - **Then:** [The expected outcome, e.g., "Service A consumes the message and creates a new user record"]

   #### Request/Message Payload
   ```json
   {{
     "//": "Example of a valid/invalid payload for this test case",
     "userId": "user-abc-123",
     "email": "test@example.com"
   }}
   ```

   #### Expected Response/Assertions
   - **Status Code:** [e.g., 201 Created (for API) or N/A (for event)]
   - **Response Body/Schema:** [e.g., Matches the defined success response schema]
   - **Target State Change:** [e.g., A new row exists in the `users` table in Service A's database]
   - **Headers/Metadata:** [e.g., `Content-Type` is `application/json`]

   #### Error Scenario Details (if applicable)
   - **Fault:** [e.g., "Malformed JSON payload", "Network timeout between services"]
   - **Expected Handling:** [e.g., "Message is sent to Dead Letter Queue", "Source service retries 3 times"]

3. Technical Test Cases:
   Focus: System limitations, infrastructure, and technical edge cases (e.g., rate limiting, concurrency, deadlocks, recovery, timeouts, performance, security).
   Documentation Requirements:
   - Higher weightage on Technical Documentation and Code. If missing, the result's precision will be lower.
   Template:
   ### Test Case TECH_[Number]
   **Technical Area:** [Performance/Security/Resilience]
   **Focus:** [e.g., Rate Limiting, Deadlock, Retry Policy]

   #### Objective
   [e.g., "To determine the maximum throughput of the `orders` service before latency exceeds 500ms."]

   #### Test Hypothesis
   [e.g., "The system will gracefully handle traffic up to 1000 requests/sec by returning 429 Too Many Requests without crashing."]

   #### Test Setup
   - **Target Component(s):** [List of services/infrastructure under test]
   - **Tooling:** [e.g., k6, JMeter, chaos-mesh]
   - **Monitoring:** [e.g., Prometheus, Grafana dashboards to watch]
   - **Load Profile/Attack Vector:** [Description of the load pattern or security threat]

   #### Execution Steps
   1. **Establish Baseline:** [Run a low-load test to get baseline metrics.]
   2. **Inject Load/Fault:** [Gradually increase load or inject the fault, e.g., "Increase concurrent users by 10 every 30 seconds."]
   3. **Monitor System:** [Observe key metrics during the test.]
   4. **Halt Condition:** [When to stop the test, e.g., "When p99 latency > 1s or error rate > 5%."]

   #### Success Criteria (Assertions)
   - **Performance:** [e.g., p99 latency < 500ms, CPU utilization < 80%]
   - **Error Rate:** [e.g., Must be less than 1%]
   - **System Behavior:** [e.g., System recovers within 5 minutes after load is removed. No data corruption occurs.]
   - **Security:** [e.g., The SQL injection attempt is blocked and logged.]

   #### Failure Analysis
   - **Expected Failure Mode:** [e.g., "Service scales up pods", "Requests get throttled with 429"]
   - **Unexpected Failure Mode:** [e.g., "Service crashes", "Data gets corrupted"]

4. Mocked System Tests:
   Focus: Isolated component testing with mocked dependencies.
   Documentation Requirements:
   - Requires Technical Documentation or Code to be effective.
   Template:
   ### Test Case MOCK_[Service]_[Number]
   **Component Under Test:** [Service Name/Version]
   **Type:** [Functional/Error/State]

   #### Objective
   [e.g., "Verify the `calculate-shipping` function correctly handles zip codes from mocked external `geolocation-service`."]

   #### Setup & Mocks
   - **System Under Test (SUT):** [The specific function/API endpoint being tested, e.g., `POST /shipping/calculate`]
   - **Mocked Dependencies:**
     - **Service:** `geolocation-service` | **Endpoint:** `GET /zip-info` | **Returns:** `{{"state": "CA", "is_remote": false}}`
     - **Service:** `db-connector` | **Function:** `save_calculation` | **Expected Call:** `with arguments(...)`
   - **Initial Data State:** [e.g., "The `products` table contains product with ID `prod-xyz`"]

   #### Trigger
   - **Action:** [e.g., "An HTTP POST request is made to the SUT's endpoint"]
   - **Input/Payload:**
     ```json
     {{
       "productId": "prod-xyz",
       "zipCode": "90210"
     }}
     ```

   #### Assertions & Verifications
   - **Return Value/Response:** [e.g., "The function should return `10.99`" or "The API should respond with 200 OK and body `{{\"price\": 10.99}}`"]
   - **Mock Interactions:**
     - **`geolocation-service`:** Was called exactly 1 time with `zipCode=90210`.
     - **`db-connector`:** Was called exactly 1 time with the correct calculation result.
   - **State Changes:** [e.g., "No changes to the database are expected."]

COVERAGE DISTRIBUTION:
Ensure a balanced and modern test distribution that prioritizes resilience and robustness.

1. Test Type Distribution (Test Pyramid):
   - **Mocked System Tests (30%):** The foundation of the pyramid. These fast, isolated tests provide the highest ROI for catching component-level bugs.
   - **Technical Tests (25%):** A large portion dedicated to non-functional requirements like performance, security, and resilience, ensuring the system is robust.
   - **Integration Tests (25%):** Fewer tests that focus on the contracts and interactions between services.
   - **End-to-End Tests (20%):** Used sparingly to verify complete, critical user journeys from start to finish.

2. Scenario & Coverage Distribution (for each test type):
   - **Negative, Corner & Edge Cases (40%):** The majority of tests should focus on how the system behaves under stress, with invalid data, and at its operational limits. This is key to building a resilient application.
   - **Happy Path Scenarios (30%):** A substantial set of tests to ensure the primary, expected functionality works correctly.
   - **Failure & Recovery Scenarios (15%):** Focused tests that simulate dependencies failing to verify that recovery mechanisms (retries, circuit breakers) work as intended.
   - **Non-Functional Scenarios (15%):** Specific tests dedicated to validating security vulnerabilities and performance SLAs.

DOCUMENTATION IMPACT & RECOMMENDATIONS:

1. For E2E Tests:
   - With Product Docs Only: Limited to user flows.
   - With Technical Docs: Enhanced with technical validation points.
   - With Code: Precise component interaction validation.
   - Recommendation: "For enhanced technical validation of E2E tests, consider adding: [missing technical docs/code]."

2. For Integration Tests:
   - With API/Event Docs: Contract-based testing is possible.
   - With Technical Design Docs: Enhanced flow validation.
   - With Code: Precise contract and implementation validation.
   - Recommendation: "For complete contract and integration coverage, consider adding: [missing technical docs/code]."

3. For Technical Tests:
   - With Technical Docs: System limitation testing is possible.
   - With Code: Precise edge case and implementation-specific validation.
   - Without Technical Docs/Code: Limited to generic scenarios.
   - Recommendation: "For deep technical and edge case coverage, ingesting [Technical Design Docs/Code] is highly recommended."

4. For Mocked Tests:
   - Required: Technical Docs or Code.
   - Without Either: Cannot be generated effectively.
   - Recommendation: "Mocked System Tests require technical documentation (e.g., API Contracts, Design Docs) or direct code access to be generated."

OUTPUT STRUCTURE:
1. Documentation Analysis
   - Available Documentation Types
   - Missing Critical Documents
   - Impact on Test Coverage

2. Test Case Distribution
   - End-to-End Test Cases: [Count]
   - Integration Test Cases: [Count]
   - Technical Test Cases: [Count]
   - Mocked System Tests: [Count]
   - Total Test Cases: [Count]

3. Coverage Analysis
   - Scenario Distribution
   - Technical Depth
   - Missing Coverage Areas

4. Recommendations
   - Required Documentation for Better Coverage
   - Coverage Improvement Suggestions
   - Technical Enhancement Suggestions

Generate the test cases now, following all templates and requirements exactly.
"""

# Provider-specific prompt refinements
PROVIDER_PROMPT_REFINEMENTS = {
    "llama": {
        "instruction_style": "direct",
        "complexity_reduction": True,
        "max_template_examples": 2,
        "simplified_language": True,
        "additional_guidance": """
IMPORTANT FOR LLAMA: Keep responses focused and concise. 
- Generate 2-3 test cases maximum per request
- Use simple, direct language
- Focus on one test type at a time
- Avoid overly complex nested structures
"""
    },

    "openai": {
        "instruction_style": "structured",
        "json_emphasis": True,
        "systematic_approach": True,
        "additional_guidance": """
OPENAI OPTIMIZATION: Leverage structured thinking and JSON formatting.
- Use systematic step-by-step approach
- Emphasize consistent JSON structure for test data
- Include clear validation criteria
- Provide detailed technical specifications
"""
    },

    "claude": {
        "instruction_style": "detailed",
        "context_awareness": True,
        "reasoning_emphasis": True,
        "additional_guidance": """
CLAUDE OPTIMIZATION: Leverage analytical and contextual strengths.
- Provide comprehensive context analysis before test generation
- Include detailed reasoning for test case selection
- Emphasize edge cases and error scenarios
- Use sophisticated technical language and concepts
"""
    },

    "gemini": {
        "instruction_style": "balanced",
        "multimodal_ready": True,
        "code_integration": True,
        "additional_guidance": """
GEMINI OPTIMIZATION: Balanced approach with code integration.
- Integrate code examples with test cases
- Balance technical depth with clarity
- Use standard testing terminology
- Include both positive and negative test scenarios
"""
    }
}


def get_refined_prompt(base_prompt: str, provider: str, context: str, query: str) -> str:
    """
    Apply provider-specific refinements to the base prompt.

    Args:
        base_prompt: The base prompt template
        provider: LLM provider name (gemini, openai, claude, llama)
        context: Context information
        query: User query

    Returns:
        Refined prompt optimized for the specific provider
    """
    # Get provider-specific refinements, default to gemini if provider not found
    refinements = PROVIDER_PROMPT_REFINEMENTS.get(
        provider.lower(), PROVIDER_PROMPT_REFINEMENTS["gemini"])

    # Apply provider-specific modifications
    refined_prompt = base_prompt

    # Add provider-specific guidance
    if "additional_guidance" in refinements:
        refined_prompt = f"{refinements['additional_guidance']}\n\n{refined_prompt}"

    # Apply complexity reduction for Llama
    if provider.lower() == "llama" and refinements.get("complexity_reduction"):
        refined_prompt = _apply_llama_simplification(refined_prompt)

    # Apply OpenAI JSON emphasis
    if provider.lower() == "openai" and refinements.get("json_emphasis"):
        refined_prompt = _apply_openai_json_emphasis(refined_prompt)

    # Apply Claude reasoning emphasis
    if provider.lower() == "claude" and refinements.get("reasoning_emphasis"):
        refined_prompt = _apply_claude_reasoning_emphasis(refined_prompt)

    # Fill in the template variables
    return refined_prompt.format(context=context, query=query)


def _apply_llama_simplification(prompt: str) -> str:
    """Simplify prompt for Llama model."""
    # Add simplified instructions at the beginning
    simplification = """
SIMPLIFIED INSTRUCTIONS FOR EFFICIENCY:
- Focus on generating 2-3 high-quality test cases
- Use clear, straightforward language
- Prioritize the most critical test scenarios
- Keep templates concise but complete

"""
    return simplification + prompt


def _apply_openai_json_emphasis(prompt: str) -> str:
    """Add JSON structure emphasis for OpenAI."""
    json_emphasis = """
JSON STRUCTURE EMPHASIS:
- Ensure all test data is in valid JSON format
- Use consistent field naming conventions
- Include JSON schema validation in assertions
- Structure responses systematically

"""
    return json_emphasis + prompt


def _apply_claude_reasoning_emphasis(prompt: str) -> str:
    """Add reasoning emphasis for Claude."""
    reasoning_emphasis = """
ANALYTICAL APPROACH:
- Begin with thorough documentation analysis
- Explain reasoning behind each test case selection
- Consider system interdependencies and edge cases
- Provide comprehensive coverage rationale

"""
    return reasoning_emphasis + prompt


def get_test_case_generation_prompt(provider: str, context: str, query: str) -> str:
    """
    Get the test case generation prompt optimized for the specific LLM provider.

    Args:
        provider: LLM provider name (gemini, openai, claude, llama)
        context: Context information from ingested documents
        query: User's test generation query

    Returns:
        Provider-optimized prompt ready for LLM consumption
    """
    return get_refined_prompt(TEST_CASE_GENERATION_PROMPT_TEMPLATE, provider, context, query)
