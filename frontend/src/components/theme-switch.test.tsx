// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { Providers } from "@/app/providers";
import { ThemeSwitch } from "@/components/theme-switch";

afterEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
});

describe("ThemeSwitch", () => {
  it("persists a selected light theme", async () => {
    render(<Providers><ThemeSwitch /></Providers>);

    const trigger = screen.getByRole("button", { name: /color theme: dark/i });
    fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false });
    fireEvent.click(await screen.findByRole("menuitem", { name: "Light" }));

    await waitFor(() => {
      expect(document.documentElement).toHaveAttribute("data-theme", "light");
      expect(localStorage.getItem("theme")).toBe("light");
    });
  });
});
