package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func main() {
	endpoint := os.Getenv("MCP_SERVER")
	if endpoint == "" {
		endpoint = "http://localhost:8080"
	}

	ctx := context.Background()
	client := mcp.NewClient(&mcp.Implementation{Name: "tools-list-client", Version: "v1"}, nil)
	transport := &mcp.SSEClientTransport{Endpoint: endpoint}
	session, err := client.Connect(ctx, transport, nil)
	if err != nil {
		log.Fatalf("connect failed: %v", err)
	}
	defer session.Close()

	res, err := session.ListTools(ctx, &mcp.ListToolsParams{})
	if err != nil {
		log.Fatalf("list tools failed: %v", err)
	}

	fmt.Printf("Tools (%d)\n", len(res.Tools))
	for _, t := range res.Tools {
		fmt.Printf("- %s: %s\n", t.Name, t.Description)
		// Print input schema (JSON Schema) if available
		if t.InputSchema != nil {
			b, err := json.MarshalIndent(t.InputSchema, "  ", "  ")
			if err != nil {
				fmt.Printf("  Input schema: <failed to marshal: %v>\n", err)
			} else {
				fmt.Printf("  Input schema:\n%s\n", string(b))
			}
		} else {
			fmt.Println("  Input schema: <none>")
		}
	}
}
