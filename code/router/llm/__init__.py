"""Model access: the Gemini REST client and the judge's prompt construction.

Kept in its own package so every network call, cache read, and retry lives behind
one boundary - the routing logic above it never talks to an API directly.
"""
