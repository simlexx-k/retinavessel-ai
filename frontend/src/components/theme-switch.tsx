"use client";

import { Check, Monitor, Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { DropdownMenu } from "radix-ui";
import { useSyncExternalStore } from "react";

const themeOptions = [
  { value: "dark", label: "Dark", icon: Moon },
  { value: "light", label: "Light", icon: Sun },
  { value: "system", label: "System", icon: Monitor },
] as const;

const subscribe = () => () => undefined;

export function ThemeSwitch() {
  const mounted = useSyncExternalStore(subscribe, () => true, () => false);
  const { theme, setTheme } = useTheme();
  const activeTheme = mounted ? theme ?? "dark" : "dark";
  const ActiveIcon = themeOptions.find((option) => option.value === activeTheme)?.icon ?? Moon;

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button className="icon-button theme-trigger" type="button" aria-label={`Color theme: ${activeTheme}`}>
          <ActiveIcon size={17} aria-hidden="true" />
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className="theme-menu" align="end" sideOffset={8}>
          <DropdownMenu.Label className="theme-menu-label">Interface theme</DropdownMenu.Label>
          {themeOptions.map(({ value, label, icon: Icon }) => (
            <DropdownMenu.Item className="theme-menu-item" key={value} onSelect={() => setTheme(value)}>
              <Icon size={15} aria-hidden="true" />
              <span>{label}</span>
              {activeTheme === value && <Check className="theme-check" size={14} aria-hidden="true" />}
            </DropdownMenu.Item>
          ))}
          <DropdownMenu.Arrow className="theme-menu-arrow" />
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
