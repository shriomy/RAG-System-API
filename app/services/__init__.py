"""Service layer — where all business logic lives.

Rules that hold throughout this package:

* Routes contain no logic; they validate input and delegate to a service.
* Graph nodes contain no logic; they call exactly one service and merge the
  result into AgentState.
* Services never touch the database directly — they go through repositories.
* Services depend on domain ports, not on concrete adapters.
"""
