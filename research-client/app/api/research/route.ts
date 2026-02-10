import { NextRequest, NextResponse } from "next/server";

// CRITICAL: Force the Edge Runtime for streaming
export const runtime = "edge";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "";

export async function POST(request: NextRequest) {
  if (!BACKEND_URL) {
    return NextResponse.json(
      { error: "Backend URL not configured" },
      { status: 500 }
    );
  }

  try {
    const body = await request.json();
    const { query } = body;

    if (!query || typeof query !== "string") {
      return NextResponse.json(
        { error: "Query is required and must be a string" },
        { status: 400 }
      );
    }

    // Fetch from backend (streaming)
    const response = await fetch(`${BACKEND_URL}/research`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });

    // Check response status
    if (!response.ok) {
      const errorText = await response.text();
      return NextResponse.json(
        { error: `Backend error: ${response.status} ${errorText}` },
        { status: response.status }
      );
    }

    // CRITICAL: Pipe the stream directly - do NOT use await response.json()
    return new NextResponse(response.body, {
      headers: {
        "Content-Type": "application/x-ndjson",
      },
    });

  } catch (error: any) {
    return NextResponse.json(
      { error: `Network error: ${error.message}` },
      { status: 500 }
    );
  }
}